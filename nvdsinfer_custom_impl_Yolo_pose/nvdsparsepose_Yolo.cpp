#include <algorithm>  // For standard algorithms like min and max
#include "nvdsinfer_custom_impl.h"  // NVIDIA DeepStream custom inference implementation
#include "utils.h"  // Utility functions

// Function prototype for parsing YOLO pose with Non-Maximum Suppression (NMS)
extern "C" bool NvDsInferParseYoloPoseNms(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo, 
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, 
    std::vector<NvDsInferInstanceMaskInfo>& objectList
);

// Function to extract keypoint (pose) proposals from model output
static void addPoseProposal(
    const float* output, 
    const uint& channelsSize, 
    const uint& netW, 
    const uint& netH, 
    const uint& b, 
    NvDsInferInstanceMaskInfo& bbi
) {
    uint kptsSize = channelsSize - 6;  // Number of keypoints (excluding bounding box and class info)
    bbi.mask = new float[kptsSize];  // Allocate memory for keypoint data
    for (uint p = 0; p < kptsSize / 3; ++p) {
        bbi.mask[p * 3 + 0] = clamp(output[b * channelsSize + p * 3 + 6], 0, netW);
        bbi.mask[p * 3 + 1] = clamp(output[b * channelsSize + p * 3 + 7], 0, netH);
        bbi.mask[p * 3 + 2] = output[b * channelsSize + p * 3 + 8];
    }
    bbi.mask_width = netW;  // Set mask width
    bbi.mask_height = netH;  // Set mask height
    bbi.mask_size = sizeof(float) * kptsSize;  // Set total mask size in bytes
}

// Function to convert raw bounding box values into structured format
static NvDsInferInstanceMaskInfo convertBBox(
    const float& bx1, 
    const float& by1, 
    const float& bx2, 
    const float& by2, 
    const uint& netW, 
    const uint& netH
) {
    NvDsInferInstanceMaskInfo b;

    float x1 = clamp(bx1, 0, netW);
    float y1 = clamp(by1, 0, netH);
    float x2 = clamp(bx2, 0, netW);
    float y2 = clamp(by2, 0, netH);

    b.left = x1;  // Assign left coordinate
    b.width = clamp(x2 - x1, 0, netW);  // Compute width
    b.top = y1;  // Assign top coordinate
    b.height = clamp(y2 - y1, 0, netH);  // Compute height

    return b;  // Return structured bounding box info
}

// Function to add bounding box information to a detected object proposal
static void addBBoxProposal(
    const float bx1, 
    const float by1, 
    const float bx2, 
    const float by2, 
    const uint& netW, 
    const uint& netH,
    const int maxIndex, 
    const float maxProb, 
    NvDsInferInstanceMaskInfo& bbi
) {
    bbi = convertBBox(bx1, by1, bx2, by2, netW, netH);  // Convert raw bounding box coordinates

    // Ignore bounding boxes that are too small
    if (bbi.width < 1 || bbi.height < 1) {
        return;
    }

    bbi.detectionConfidence = maxProb;  // Store detection confidence
    bbi.classId = maxIndex;  // Store detected class ID
}

// Function to decode YOLO pose tensor output into bounding boxes and keypoints
static std::vector<NvDsInferInstanceMaskInfo> decodeTensorYoloPose(
    const float* output, 
    const uint& outputSize, 
    const uint& channelsSize, 
    const uint& netW,
    const uint& netH, 
    const std::vector<float>& preclusterThreshold
) {
    std::vector<NvDsInferInstanceMaskInfo> binfo;  // Vector to store detected objects

    for (uint b = 0; b < outputSize; ++b) {  // Loop through detected objects
        float maxProb = output[b * channelsSize + 4];  // Get object confidence score
        int maxIndex = (int) output[b * channelsSize + 5];  // Get object class ID

        // Only keep detections above the confidence threshold
        if (maxProb > preclusterThreshold[0]) {
            float bx1 = output[b * channelsSize + 0];
            float by1 = output[b * channelsSize + 1];
            float bx2 = output[b * channelsSize + 2];
            float by2 = output[b * channelsSize + 3];

            NvDsInferInstanceMaskInfo bbi;  // Object information struct

            addBBoxProposal(bx1, by1, bx2, by2, netW, netH, maxIndex, maxProb, bbi);  // Add bounding box
            addPoseProposal(output, channelsSize, netW, netH, b, bbi);  // Add keypoints

            binfo.push_back(bbi);  // Store detected object info
        }
    }

    return binfo;  // Return list of detected objects
}

// Custom parsing function for YOLO pose output
static bool NvDsInferParseCustomYoloPose(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo, 
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferInstanceMaskInfo>& objectList
) {
    // Ensure there is at least one output layer
    if (outputLayersInfo.empty()) {
        std::cerr << "ERROR: Could not find output layer in bbox parsing" << std::endl;
        return false;
    }

    const NvDsInferLayerInfo& output = outputLayersInfo[0];  // Get first output layer

    const uint outputSize = output.inferDims.d[0];  // Number of detected objects
    const uint channelsSize = output.inferDims.d[1];  // Number of output channels per object

    // Decode YOLO pose tensor output
    objectList = decodeTensorYoloPose(
        (const float*) (output.buffer), 
        outputSize, 
        channelsSize, 
        networkInfo.width, 
        networkInfo.height, 
        detectionParams.perClassPreclusterThreshold
    );

    return true;
}

// Function wrapper for YOLO pose parsing with NMS
extern "C" bool NvDsInferParseYoloPoseNms(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo, 
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, 
    std::vector<NvDsInferInstanceMaskInfo>& objectList
) {
    return NvDsInferParseCustomYoloPose(outputLayersInfo, networkInfo, detectionParams, objectList);
}

// Macro to check the function prototype for custom instance mask parsing
CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloPoseNms);
