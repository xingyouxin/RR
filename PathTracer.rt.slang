/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/

/** Path tracing pass.

    This file contains the entry points for all ray tracing programs.
    We import the path tracer utility functions defined in PathTracer.slang.

    The host sets the compile-time constants in StaticParams.slang.
    It also sets the following defines for optional I/O buffers:

    is_valid_<name> is 1 if buffer with this name is bound, 0 otherwise.
*/
import PathTracer;
import Utils.Timing.GpuTimer;
import RenderPasses.Shared.PathTracer.LoadShadingData;
import BRDFViewerParams; // For UI.

ParameterBlock<PathTracerData> gData;

cbuffer PerFrameCB
{
    BRDFViewerParams gParams;
}

struct QuadTreeNodeCMP
{
    half4 coeff1 = half4(0.f);
    half4 coeff2 = half4(0.f);
    half4 coeff3 = half4(0.f);
    half4 child = half4(0.f);
};

Buffer<uint> gQuadTreeBRDFBufferIndex;   // quad tree test.
StructuredBuffer<QuadTreeNodeCMP> gQuadTreeBRDF[5];     // quad tree test.
Buffer<uint> gQuadTreeBRDFIndex; // quad tree test.

Buffer<uint> gQuadTreeLightBufferIndex;   // quad tree test.
StructuredBuffer<QuadTreeNodeCMP> gQuadTreeLight[5];     // quad tree test.
Buffer<uint> gQuadTreeLightIndex;// quad tree test.

// Outputs (optional)
RWTexture2D<float4> gOutputColor;
RWTexture2D<float4> gOutputAlbedo;
RWTexture2D<uint> gOutputTime;
RWTexture2D<float4> gOutputTimeTest;

// Static configuration based on which buffers are bound.
#define isValid(name) (is_valid_##name != 0)

#define PI 3.14159265358979323846

#define INV_SQRT_TWO  0.70710678118654752440f

#define QuadTreeProductWithInterpolation
//#define QuadTreeProductWithoutInterpolation

#define woResTimes 3

/** ********************* Ray index 0: Scatter ray ************************ */

[shader("miss")]
void scatterMiss(inout ScatterRayData rayData : SV_RayPayload)
{
}

[shader("anyhit")]
void scatterAnyHit(
    uniform HitShaderParams hitParams,
    inout ScatterRayData rayData : SV_RayPayload, BuiltInTriangleIntersectionAttributes attribs : SV_IntersectionAttributes)
{
#if USE_ALPHA_TEST
    // Alpha test for non-opaque geometry.
    VertexData v = getVertexData(hitParams, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(hitParams.getGlobalHitID());
    if (alphaTest(v, gScene.materials[materialID], gScene.materialResources[materialID], 0.f)) IgnoreHit();
#endif
}

[shader("closesthit")]
void scatterClosestHit(
    uniform HitShaderParams hitParams,
    inout ScatterRayData rayData : SV_RayPayload, BuiltInTriangleIntersectionAttributes attribs : SV_IntersectionAttributes)
{
    // Store hit information. Note we don't access the materials here.
    HitInfo hit;
    hit.type = InstanceType::TriangleMesh;
    hit.instanceID = hitParams.getGlobalHitID();
    hit.primitiveIndex = PrimitiveIndex();
    hit.barycentrics = attribs.barycentrics;
    rayData.packedHitInfo = hit.encode();
}

/************************** Ray index 1: Shadow ray ************************ */

[shader("miss")]
void shadowMiss(inout ShadowRayData rayData : SV_RayPayload)
{
    // The miss shader is executed if the ray misses all geometry. Mark as visible.
    rayData.visible = true;
}

[shader("anyhit")]
void shadowAnyHit(
    uniform HitShaderParams hitParams,
    inout ShadowRayData rayData : SV_RayPayload, BuiltInTriangleIntersectionAttributes attribs : SV_IntersectionAttributes)
{
#if USE_ALPHA_TEST
    // Alpha test for non-opaque geometry.
    VertexData v = getVertexData(hitParams, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(hitParams.getGlobalHitID());
    if (alphaTest(v, gScene.materials[materialID], gScene.materialResources[materialID], 0.f)) IgnoreHit();
#endif
}

/** ******************************** RayGen ******************************** */

/** This is the entry point for the path tracer.

    We generate N paths (= #spp) per pixel, which are traced into the scene.
    The path tracer is written as a for-loop over path segments, where each
    iteration traces a shadow ray for direct illumination and a scatter ray.

    The hit shader for the scatter ray currently generates ray parameters for
    the shadow ray to evaluate direct illumination and generates ray parameters
    for the next scatter ray, which are both returned the raygen shader to be
    traced. This is more efficient than tracing from the hit shader. The max
    recusion depth = 1.
*/

//以下为GPU版本
static const int MAXQSIZE = 955;
//循环队列
struct my_queue
{
    int front = 0;
    int rear = 0;
    uint base[MAXQSIZE];
    int count = 0;
};
static my_queue qu;

void init_queue()
{
    qu.front = 0;
    qu.rear = 0;
    qu.count = 0;
}

void qu_push(uint x)
{
    qu.base[qu.rear] = x;
    qu.rear = (qu.rear + 1) % MAXQSIZE;//optimizable WAIT
    qu.count++;
}

float qu_pop()
{
    float x = qu.base[qu.front];
    qu.front = (qu.front + 1) % MAXQSIZE;//optimizable WAIT
    qu.count--;
    return x;
}

bool qu_empty()
{
    return (qu.front == qu.rear && qu.count == 0);
}

bool qu_full()
{
    return (qu.front == qu.rear && qu.count == MAXQSIZE);
}

#ifdef QuadTreeProductWithInterpolation

//0~2：avg  3~5：coeff[0]  6~8：coeff[1]  9~11：coeff[2] 12~15：孩子指针
float3 Tree_convolution_gpu(uint4 brdfBId, uint4 brdfSId, uint lightBId, uint lightSId, float wZ, float wPhi)
{
    //my_queue qu;
    float3 col = float3(0.f);
    uint root = 0; //0 * node_datasize
    init_queue();
    qu_push(root);
    qu_push(root);
    qu_push(root);
    qu_push(root);
    qu_push(root);

    while (!qu_empty())
    {
        // brdf*4.
        uint4 node_index1 = uint4(0); 
        node_index1[0] = brdfSId[0] + qu.base[qu.front];//optimizable LGSB
        qu_pop();
        node_index1[1] = brdfSId[1] + qu.base[qu.front];//optimizable LGSB
        qu_pop();
        node_index1[2] = brdfSId[2] + qu.base[qu.front];//optimizable LGSB
        qu_pop();
        node_index1[3] = brdfSId[3] + qu.base[qu.front];
        qu_pop();

        // light*1.
        uint node_index2 = lightSId + qu.base[qu.front];

        //lightBId=0;
        //uint node_index2 = qu.base[qu.front];
        qu_pop();
        
        bool calculate_flag = false;
        bool cell_flag[4] = {false};
        float3 child_col[5][4];
        bool all_null_flag = true;

        QuadTreeNodeCMP BRDFQTNode[4];
        BRDFQTNode[0] = gQuadTreeBRDF[brdfBId[0]][node_index1[0]];
        BRDFQTNode[1] = gQuadTreeBRDF[brdfBId[1]][node_index1[1]];
        BRDFQTNode[2] = gQuadTreeBRDF[brdfBId[2]][node_index1[2]];
        BRDFQTNode[3] = gQuadTreeBRDF[brdfBId[3]][node_index1[3]];

        QuadTreeNodeCMP LightQTNode = gQuadTreeLight[lightBId][node_index2];

        for (int i = 0; i < 4; i++)
        {
            // brdf*4.
            if (//optimizable LGSB
                (int)BRDFQTNode[0].child[i] == -1
             || (int)BRDFQTNode[1].child[i] == -1
             || (int)BRDFQTNode[2].child[i] == -1
             || (int)BRDFQTNode[3].child[i] == -1
             // light.
             || (int)LightQTNode.child[i] == -1)
            {
                cell_flag[i] = true;
                calculate_flag = true;
            }
            else
            {
                all_null_flag = false; // 当前节点不是叶子节点，存在一个非叶子节点孩子
            }
        }
        
        if (calculate_flag && all_null_flag == false)//有NULL但不为全NULL
        {
            float3 coeff_1 = float3(0.f);
            float3 coeff_2 = float3(0.f);
            float3 coeff_3 = float3(0.f);
            float3 avg = float3(0.f);

            float3 a_1 = float3(0.f);
            float3 a_2 = float3(0.f);
            float3 d_1 = float3(0.f);
            float3 d_2 = float3(0.f);
            
            // brdf*4.
            for(int i = 0; i < 4; i++)
            {
                // mark 1 
                coeff_1 = BRDFQTNode[i].coeff1.xyz;
                coeff_2 = BRDFQTNode[i].coeff2.xyz;
                coeff_3 = BRDFQTNode[i].coeff3.xyz;
                avg = float3(BRDFQTNode[i].coeff1.w, BRDFQTNode[i].coeff2.w, BRDFQTNode[i].coeff3.w);

                //a_1 = (avg + coeff_1) * INV_SQRT_TWO;
                //a_2 = (avg - coeff_1) * INV_SQRT_TWO;
                //d_1 = (coeff_2 + coeff_3) * INV_SQRT_TWO;
                //d_2 = (coeff_2 - coeff_3) * INV_SQRT_TWO;
                //
                //child_col[i][0] = (a_1 + d_1) * INV_SQRT_TWO;
                //child_col[i][1] = (a_1 - d_1) * INV_SQRT_TWO;
                //child_col[i][2] = (a_2 + d_2) * INV_SQRT_TWO;
                //child_col[i][3] = (a_2 - d_2) * INV_SQRT_TWO;

                a_1 = avg + coeff_1;
                a_2 = avg - coeff_1;
                d_1 = coeff_2 + coeff_3;
                d_2 = coeff_2 - coeff_3;

                child_col[i][0] = (a_1 + d_1) * 0.5f;
                child_col[i][1] = (a_1 - d_1) * 0.5f;
                child_col[i][2] = (a_2 + d_2) * 0.5f;
                child_col[i][3] = (a_2 - d_2) * 0.5f;
            }

            // light.
            coeff_1 = LightQTNode.coeff1.xyz;
            coeff_2 = LightQTNode.coeff2.xyz;
            coeff_3 = LightQTNode.coeff3.xyz;
            avg = float3(LightQTNode.coeff1.w, LightQTNode.coeff2.w, LightQTNode.coeff3.w);

            //a_1 = (avg + coeff_1) * INV_SQRT_TWO;
            //a_2 = (avg - coeff_1) * INV_SQRT_TWO;
            //d_1 = (coeff_2 + coeff_3) * INV_SQRT_TWO;
            //d_2 = (coeff_2 - coeff_3) * INV_SQRT_TWO;
            //
            //child_col[4][0] = (a_1 + d_1) * INV_SQRT_TWO;
            //child_col[4][1] = (a_1 - d_1) * INV_SQRT_TWO;
            //child_col[4][2] = (a_2 + d_2) * INV_SQRT_TWO;
            //child_col[4][3] = (a_2 - d_2) * INV_SQRT_TWO;

            a_1 = avg + coeff_1;
            a_2 = avg - coeff_1;
            d_1 = coeff_2 + coeff_3;
            d_2 = coeff_2 - coeff_3;

            child_col[4][0] = (a_1 + d_1) * 0.5f;
            child_col[4][1] = (a_1 - d_1) * 0.5f;
            child_col[4][2] = (a_2 + d_2) * 0.5f;
            child_col[4][3] = (a_2 - d_2) * 0.5f;
        }

        
        if (all_null_flag)
        {
            // brdf bilinear interpolation. Avg
            float3 brdfAvg00 = float3(BRDFQTNode[0].coeff1.w, BRDFQTNode[0].coeff2.w, BRDFQTNode[0].coeff3.w);////optimizable SELECT
            float3 brdfAvg01 = float3(BRDFQTNode[1].coeff1.w, BRDFQTNode[1].coeff2.w, BRDFQTNode[1].coeff3.w);
            float3 brdfAvg10 = float3(BRDFQTNode[2].coeff1.w, BRDFQTNode[2].coeff2.w, BRDFQTNode[2].coeff3.w);
            float3 brdfAvg11 = float3(BRDFQTNode[3].coeff1.w, BRDFQTNode[3].coeff2.w, BRDFQTNode[3].coeff3.w);
            
            float3 brdfAvg0001 = brdfAvg00*(1-wZ) + brdfAvg01*wZ;
            float3 brdfAvg1011 = brdfAvg10*(1-wZ) + brdfAvg11*wZ;
            
            float3 brdfAvg = brdfAvg0001*(1-wPhi) + brdfAvg1011*wPhi;
            col += brdfAvg * float3(LightQTNode.coeff1.w, LightQTNode.coeff2.w, LightQTNode.coeff3.w);
            
            // brdf bilinear interpolation. Coeff*3
            // coeff1.
            float3 brdfCoe00 = BRDFQTNode[0].coeff1.xyz;
            float3 brdfCoe01 = BRDFQTNode[1].coeff1.xyz;
            float3 brdfCoe10 = BRDFQTNode[2].coeff1.xyz;
            float3 brdfCoe11 = BRDFQTNode[3].coeff1.xyz;
            
            float3 brdfCoe0001 = brdfCoe00*(1-wZ) + brdfCoe01*wZ;
            float3 brdfCoe1011 = brdfCoe10*(1-wZ) + brdfCoe11*wZ;
            
            float3 brdfCoe = brdfCoe0001*(1-wPhi) + brdfCoe1011*wPhi;
            col += brdfCoe * LightQTNode.coeff1.xyz;
            
            // coeff2.
            brdfCoe00 = BRDFQTNode[0].coeff2.xyz;
            brdfCoe01 = BRDFQTNode[1].coeff2.xyz;
            brdfCoe10 = BRDFQTNode[2].coeff2.xyz;
            brdfCoe11 = BRDFQTNode[3].coeff2.xyz;
            
            brdfCoe0001 = brdfCoe00*(1-wZ) + brdfCoe01*wZ;
            brdfCoe1011 = brdfCoe10*(1-wZ) + brdfCoe11*wZ;
            
            brdfCoe = brdfCoe0001*(1-wPhi) + brdfCoe1011*wPhi;
            col += brdfCoe * LightQTNode.coeff2.xyz;
            
            // coeff3.
            brdfCoe00 = BRDFQTNode[0].coeff3.xyz;
            brdfCoe01 = BRDFQTNode[1].coeff3.xyz;
            brdfCoe10 = BRDFQTNode[2].coeff3.xyz;
            brdfCoe11 = BRDFQTNode[3].coeff3.xyz;
            
            brdfCoe0001 = brdfCoe00*(1-wZ) + brdfCoe01*wZ;
            brdfCoe1011 = brdfCoe10*(1-wZ) + brdfCoe11*wZ;
            
            brdfCoe = brdfCoe0001*(1-wPhi) + brdfCoe1011*wPhi;
            col += brdfCoe * LightQTNode.coeff3.xyz;

        }
        else
        {
            for (int i = 0; i < 4; i++)
            {
                if (cell_flag[i])
                {
                    float3 brdf_child_col00 = child_col[0][i];
                    float3 brdf_child_col01 = child_col[1][i];
                    float3 brdf_child_col10 = child_col[2][i];
                    float3 brdf_child_col11 = child_col[3][i];

                    float3 brdf_child_col0001 = brdf_child_col00*(1-wZ) + brdf_child_col01*wZ;
                    float3 brdf_child_col1011 = brdf_child_col10*(1-wZ) + brdf_child_col11*wZ;

                    float3 brdf_child_col = brdf_child_col0001*(1-wPhi) + brdf_child_col1011*wPhi;

                    col += brdf_child_col * child_col[4][i];
                }
                else
                {
                    qu_push((uint)BRDFQTNode[0].child[i]);
                    qu_push((uint)BRDFQTNode[1].child[i]);
                    qu_push((uint)BRDFQTNode[2].child[i]);
                    qu_push((uint)BRDFQTNode[3].child[i]);
                    qu_push((uint)LightQTNode.child[i]);
                }
            }
        }
    }
    return col;
}


[shader("raygeneration")]
void rayGen()
{
    uint2 launchIndex = DispatchRaysIndex().xy;
    uint2 launchDim = DispatchRaysDimensions().xy;

    logSetPixel(launchIndex);
    printSetPixel(launchIndex);

    GpuTimer timer;
    if (isValid(gOutputTime)) timer.start();

    float3 outColor = float3(0, 0, 0);
    float3 outAlbedo = float3(0, 0, 0);
    float outAlpha = 0.f;

    HitInfo hit;
    ShadingData sd;
    if (loadShadingData(launchIndex, launchDim, gScene.camera, sd, hit))
    {
        // meshId
        uint meshId = hit.instanceID;
        // faceId
        uint faceId = hit.primitiveIndex;
        // materialId
        uint materialId = sd.materialID;

        // BRDF coe start id.
        float3 wo = normalize(toLocal(sd.V, sd));
        uint4 woId = uint4(0);
        // get woId, nouniform sampling wo.
        float z = wo.z;
        float phi = atan2(wo.y, wo.x)*180/PI;
        if(phi < 0) phi += 360.0f;
        if(wo.x == 0 && wo.y == 0)
        {
            phi = 0.f;
        }

        uint size_phi = 360*woResTimes;
        uint size_z = 57*woResTimes;

        // for bilinear interpolation.
        woId[0] = size_phi*floor(size_z*z) + floor(size_phi*phi/360.f);
        woId[1] = size_phi*ceil(size_z*z) + floor(size_phi*phi/360.f); 
        woId[2] = size_phi*floor(size_z*z) + ceil(size_phi*phi/360.f); 
        woId[3] = size_phi*ceil(size_z*z) + ceil(size_phi*phi/360.f); 
        float wZ = size_z*z - floor(size_z*z);
        float wPhi = size_phi*phi/360.f - floor(size_phi*phi/360.f);


        for(uint i = 0; i < 3; i++)
        {
            uint4 brdfCoeStartId = uint4(0);
            uint4 brdfCoeBufferId = uint4(0);

            for (uint j = 0; j < 4; j++)
            {
                brdfCoeStartId[j] = gQuadTreeBRDFIndex[woId[j]*3 + i];
            }

            uint lightCoeStartId = gQuadTreeLightIndex[faceId*3 + i];
            uint lightCoeBufferId = 0;

            // product.
            outColor += Tree_convolution_gpu(brdfCoeBufferId, brdfCoeStartId, lightCoeBufferId, lightCoeStartId, wZ, wPhi);

        }

       
        outColor = outColor/16384;

        outAlbedo = sd.diffuse + sd.specular;
        outAlpha = 1.f;
    }
    else
    {
        // Background pixel.
        outColor = evalBackground(-sd.V);
        outAlbedo = outColor.rgb;
        outAlpha = kForceAlphaOne ? 1.f : 0.f;
    }

    // DEBUG
    assert(!any(isnan(outColor)));

    // Write outputs.
    // These are all optional so using compile-time checks to decide which ones to write.
    if (isValid(gOutputColor)) gOutputColor[launchIndex] = float4(outColor, outAlpha);
    if (isValid(gOutputAlbedo)) gOutputAlbedo[launchIndex] = float4(outAlbedo, 1);

    // Write time.
    if (isValid(gOutputTime)) gOutputTime[launchIndex] = timer.getElapsed();
}

#endif


#ifdef QuadTreeProductWithoutInterpolation

//0~2：avg  3~5：coeff[0]  6~8：coeff[1]  9~11：coeff[2] 12~15：孩子指针
float3 Tree_convolution_gpu(uint brdfBId, uint brdfSId, uint lightBId, uint lightSId)
{
    //my_queue qu;
    float3 col = float3(0.f);
    uint root = 0; //0 * node_datasize
    init_queue();
    qu_push(root);
    qu_push(root);

    while (!qu_empty())
    {   
        uint node_index1 = brdfSId + qu.base[qu.front];
        qu_pop();
        uint node_index2 = lightSId + qu.base[qu.front];
        //lightBId = 0;
        //uint node_index2 = qu.base[qu.front];
        qu_pop();
        
        bool calculate_flag = false;
        bool cell_flag[4] = {false};
        float3 child_col[2][4];
        bool all_null_flag = true;

        QuadTreeNodeCMP QTNodeBRDF = gQuadTreeBRDF[brdfBId][node_index1];
        QuadTreeNodeCMP QTNodeLight = gQuadTreeLight[lightBId][node_index2];

        for (int i = 0; i < 4; i++)
        {
            if ((int)QTNodeBRDF.child[i] == -1 || (int)QTNodeLight.child[i] == -1)
            {
                cell_flag[i] = true;
                calculate_flag = true;
            }
            else
            {
                all_null_flag = false; // 当前节点不是叶子节点，存在一个非叶子节点孩子
            }
        }
        
        if (calculate_flag && all_null_flag == false)//有NULL但不为全NULL
        {
            float3 coeff_1 = float3(0.f);
            float3 coeff_2 = float3(0.f);
            float3 coeff_3 = float3(0.f);
            float3 avg = float3(0.f);

            float3 a_1 = float3(0.f);
            float3 a_2 = float3(0.f);
            float3 d_1 = float3(0.f);
            float3 d_2 = float3(0.f);

            coeff_1 = QTNodeBRDF.coeff1.xyz;
            coeff_2 = QTNodeBRDF.coeff2.xyz;
            coeff_3 = QTNodeBRDF.coeff3.xyz;
            avg = float3(QTNodeBRDF.coeff1.w, QTNodeBRDF.coeff2.w, QTNodeBRDF.coeff3.w);

            //a_1 = (avg + coeff_1) * INV_SQRT_TWO;
            //a_2 = (avg - coeff_1) * INV_SQRT_TWO;
            //d_1 = (coeff_2 + coeff_3) * INV_SQRT_TWO;
            //d_2 = (coeff_2 - coeff_3) * INV_SQRT_TWO;
            //
            //child_col[0][0] = (a_1 + d_1) * INV_SQRT_TWO;
            //child_col[0][1] = (a_1 - d_1) * INV_SQRT_TWO;
            //child_col[0][2] = (a_2 + d_2) * INV_SQRT_TWO;
            //child_col[0][3] = (a_2 - d_2) * INV_SQRT_TWO;

            a_1 = (avg + coeff_1);
            a_2 = (avg - coeff_1);
            d_1 = (coeff_2 + coeff_3);
            d_2 = (coeff_2 - coeff_3);

            child_col[0][0] = (a_1 + d_1) * 0.5f;
            child_col[0][1] = (a_1 - d_1) * 0.5f;
            child_col[0][2] = (a_2 + d_2) * 0.5f;
            child_col[0][3] = (a_2 - d_2) * 0.5f;

            coeff_1 = gQuadTreeLight[lightBId][node_index2].coeff1.xyz;
            coeff_2 = gQuadTreeLight[lightBId][node_index2].coeff2.xyz;
            coeff_3 = gQuadTreeLight[lightBId][node_index2].coeff3.xyz;
            avg = float3(QTNodeLight.coeff1.w, QTNodeLight.coeff2.w, QTNodeLight.coeff3.w);

            //a_1 = (avg + coeff_1) * INV_SQRT_TWO;
            //a_2 = (avg - coeff_1) * INV_SQRT_TWO;
            //d_1 = (coeff_2 + coeff_3) * INV_SQRT_TWO;
            //d_2 = (coeff_2 - coeff_3) * INV_SQRT_TWO;
            //
            //child_col[1][0] = (a_1 + d_1) * INV_SQRT_TWO;
            //child_col[1][1] = (a_1 - d_1) * INV_SQRT_TWO;
            //child_col[1][2] = (a_2 + d_2) * INV_SQRT_TWO;
            //child_col[1][3] = (a_2 - d_2) * INV_SQRT_TWO;

            a_1 = (avg + coeff_1);
            a_2 = (avg - coeff_1);
            d_1 = (coeff_2 + coeff_3);
            d_2 = (coeff_2 - coeff_3);
            
            child_col[1][0] = (a_1 + d_1) * 0.5f;
            child_col[1][1] = (a_1 - d_1) * 0.5f;
            child_col[1][2] = (a_2 + d_2) * 0.5f;
            child_col[1][3] = (a_2 - d_2) * 0.5f;
        }

        
        if (all_null_flag)
        {
            col += float3(QTNodeBRDF.coeff1.w, QTNodeBRDF.coeff2.w, QTNodeBRDF.coeff3.w) * float3(QTNodeLight.coeff1.w, QTNodeLight.coeff2.w, QTNodeLight.coeff3.w);
            col += QTNodeBRDF.coeff1.xyz * QTNodeLight.coeff1.xyz;
            col += QTNodeBRDF.coeff2.xyz * QTNodeLight.coeff2.xyz;
            col += QTNodeBRDF.coeff3.xyz * QTNodeLight.coeff3.xyz;
        }
        else
        {
            for (int i = 0; i < 4; i++)
            {
                if (cell_flag[i])
                {
                    col += child_col[0][i] * child_col[1][i];
                }
                else
                {   
                    qu_push((uint)QTNodeBRDF.child[i]);
                    qu_push((uint)QTNodeLight.child[i]);
                }
            }
        }
    }
    return col;
}

[shader("raygeneration")]
void rayGen()
{
    uint2 launchIndex = DispatchRaysIndex().xy;
    uint2 launchDim = DispatchRaysDimensions().xy;

    logSetPixel(launchIndex);
    printSetPixel(launchIndex);

    //GpuTimer timer;
    //if (isValid(gOutputTime)) timer.start();

    float3 outColor = float3(0, 0, 0);
    float3 outAlbedo = float3(0, 0, 0);
    float outAlpha = 0.f;

    GpuTimer timer;
    timer.start();

    HitInfo hit;
    ShadingData sd;
    if (loadShadingData(launchIndex, launchDim, gScene.camera, sd, hit))
    {
        // meshId
        uint meshId = hit.instanceID;
        // faceId
        uint faceId = hit.primitiveIndex;
        // materialId
        uint materialId = sd.materialID;

        // BRDF coe start id.
        float3 wo = normalize(toLocal(sd.V, sd));
        uint woId = 0;
        // get woId, nouniform sampling wo.
        float z = wo.z;
        float phi = atan2(wo.y, wo.x)*180/PI;
        if(phi < 0) phi += 360.0f;
        if(wo.x == 0 && wo.y == 0)
        {
            phi = 0.f;
        }

        uint size_phi = 360*woResTimes;
        uint size_z = 57*woResTimes;

        woId = size_phi*round(size_z*z) + floor(size_phi*phi/360.f);
       
        for(uint i = 0; i < 3; i++)
        {
            uint brdfCoeStartId = 0;
            uint brdfCoeBufferId = 0;
            
            brdfCoeStartId = gQuadTreeBRDFIndex[woId*3 + i];

            uint lightCoeStartId = gQuadTreeLightIndex[faceId*3 + i];
            uint lightCoeBufferId = 0;

            // product.
            outColor += Tree_convolution_gpu(brdfCoeBufferId, brdfCoeStartId, lightCoeBufferId, lightCoeStartId);
            //outColor = float3(1.f);
            //outColor += float3(brdfCoeStartId, lightCoeStartId, 0.f);

        }

        //outColor = outColor/(size_phi*(size_z+1) + 16384);
        outColor = outColor/16384;

        outAlbedo = sd.diffuse + sd.specular;
        outAlpha = 1.f;
    }
    else
    {
        // Background pixel.
        outColor = evalBackground(-sd.V);
        outAlbedo = outColor.rgb;
        outAlpha = kForceAlphaOne ? 1.f : 0.f;
    }

    // DEBUG
    assert(!any(isnan(outColor)));

    // Write outputs.
    // These are all optional so using compile-time checks to decide which ones to write.
    if (isValid(gOutputColor)) gOutputColor[launchIndex] = float4(outColor, outAlpha);
    if (isValid(gOutputAlbedo)) gOutputAlbedo[launchIndex] = float4(outAlbedo, 1);
    // Write time.
    if (isValid(gOutputTimeTest)) gOutputTimeTest[launchIndex] = timer.getElapsed()/(pow(10.f, 6));
}


#endif
