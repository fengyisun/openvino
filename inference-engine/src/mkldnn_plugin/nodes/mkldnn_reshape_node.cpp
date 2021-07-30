// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_reshape_node.h"
#include <string>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include <ngraph/opsets/opset1.hpp>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNReshapeNode::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (isDynamicNgraphNode(op)) {
            errorMessage = "Doesn't support op with dynamic shapes";
            return false;
        }
        if (!std::dynamic_pointer_cast<const ngraph::opset1::Reshape>(op) &&
            !std::dynamic_pointer_cast<const ngraph::opset1::Squeeze>(op) &&
                !std::dynamic_pointer_cast<const ngraph::opset1::Unsqueeze>(op)) {
            errorMessage = "Only opset1 Reshape, Squeeze, Unsqueeze operations are supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNReshapeNode::MKLDNNReshapeNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache) :
        MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }
}

MKLDNNReshapeNode::MKLDNNReshapeNode(const std::string& name, const Shape& inDims, const Shape& outDims, Precision precision,
        const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &wCache)
        : MKLDNNNode("Reshape", name, eng, wCache) {
    this->inputShapes.push_back(inDims);
    this->outputShapes.push_back(outDims);
    addOriginalInputPrecision(precision);
    addOriginalOutputPrecision(precision);
}

void MKLDNNReshapeNode::getSupportedDescriptors() {
    if (getParentEdges().size() != 1 && getParentEdges().size() != 2)
        IE_THROW() << "Incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        IE_THROW() << "Incorrect number of output edges for layer " << getName();
}

std::vector<mkldnn::memory::format_tag> MKLDNNReshapeNode::getDataFormats(const int ndims) const {
    switch (ndims) {
        case 1:
            return {memory::format_tag::a};
        case 2:
            return {memory::format_tag::ab, memory::format_tag::ba};
        case 3:
            return {memory::format_tag::abc, memory::format_tag::acb};
        case 4:
            return {memory::format_tag::abcd, memory::format_tag::acdb};
        case 5:
            return {memory::format_tag::abcde};
        case 6:
            return {memory::format_tag::abcdef};
        defalut:
            return {memory::format_tag::undef};
    }
    return {memory::format_tag::undef};
}

void MKLDNNReshapeNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    InferenceEngine::Precision inPrec = getOriginalInputPrecisionAtPort(0);
    InferenceEngine::Precision outPrec = getOriginalOutputPrecisionAtPort(0);

    // Current reshape implementation is simple memory reinterpret,
    // same precision on input and output is required
    if (inPrec != outPrec)
        inPrec = outPrec;

    NodeConfig config;
    auto parent = getParentEdgeAt(0)->getParent();
    auto prSupportedDesc = parent->getSupportedPrimitiveDescriptors();
    auto outDims = getChildEdgeAt(0)->getDims().ndims();
    for (auto prDescInfo : prSupportedDesc) {
        auto inNum = getParentEdgeAt(0)->getInputNum();
        auto prOutConfs = prDescInfo.getConfig().outConfs[inNum];
        auto inLayout = prOutConfs.desc.getLayout();
        auto inFmt = MKLDNNMemory::Convert(inLayout);
        for (auto outFmt : getDataFormats(outDims)) {
            InferenceEngine::LayerConfig config;
            config.dynBatchSupport = true;
            config.inConfs.resize(getParentEdges().size());
            for (int i = 0; i < getParentEdges().size(); i++) {
                config.inConfs[i].inPlace = -1;
                config.inConfs[i].constant = false;
                if (i == 0)
                    config.inConfs[0].desc = MKLDNNMemoryDesc(MKLDNNDims(prOutConfs.desc.getDims()), inputDataType, inFmt);
                else
                    config.inConfs[i].desc = MKLDNNMemoryDesc(getParentEdgeAt(i)->getDims(), inputDataType);
            }
            config.outConfs.resize(1);
            config.outConfs[0].inPlace = 0;
            config.outConfs[0].constant = false;
            config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, outFmt);
            supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);
        }
    }
}

void MKLDNNReshapeNode::createPrimitive() {
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        IE_THROW() << "Destination memory didn't allocate.";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        IE_THROW() << "Input memory didn't allocate.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        IE_THROW() << "Preferable primitive descriptor is not set.";
}

bool MKLDNNReshapeNode::created() const {
    return getType() == Reshape;
}
REG_MKLDNN_PRIM_FOR(MKLDNNReshapeNode, Reshape);
