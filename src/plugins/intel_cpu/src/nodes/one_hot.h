// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <node.h>
#include <string>
#include <memory>
#include <vector>
#include <ie_blob.h>

namespace ov {
namespace intel_cpu {
namespace node {

class OneHot : public Node {
public:
    OneHot(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, WeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override {};
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    bool needShapeInfer() const override;
    std::vector<VectorDims> shapeInfer() const override;
    bool needPrepareParams() const override { return false; };
    void executeDynamicImpl(mkldnn::stream strm) override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    typedef InferenceEngine::PrecisionTrait<InferenceEngine::Precision::I32>::value_type in_type;

    struct OneHotContext {
        OneHot* nodePtr;
        size_t prefix_size;
        size_t suffix_size;
    };

    template<typename dst_t>
    struct OneHotExecute {
        void operator()(OneHotContext & ctx) {
            ctx.nodePtr->one_hot<dst_t>(ctx.prefix_size, ctx.suffix_size);
        }
    };

    mutable Dim depth = Shape::UNDEFINED_DIM;
    int32_t axis = -1;

    InferenceEngine::Precision output_precision;

    std::string errorPrefix;

    static const size_t INDICES_ID = 0;
    static const size_t DEPTH_ID = 1;
    static const size_t ON_VALUE_ID = 2;
    static const size_t OFF_VALUEAXES_ID = 3;

    template<typename out_type>
    void one_hot(size_t prefix_size, size_t suffix_size);
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
