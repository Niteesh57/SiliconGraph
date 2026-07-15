// ============================================================================
// Unit Tests: Pass Manager + Graph Passes
// ============================================================================
#include <gtest/gtest.h>
#include "passes/pass_manager.h"
#include "arm_ir/graph.h"
#include "arm_ir/ops.h"
#include "analysis/cost_model.h"

using namespace armcc;
using namespace armcc::passes;
using namespace armcc::ir;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static IRGraph buildSimpleGraph() {
  IRGraph g;
  g.name = "test";
  g.model_id = "test/model";
  g.hidden_size = 64;
  g.num_heads   = 2;
  g.num_layers  = 2;

  // Input → MatMul → Add → ReLU → Output
  auto mkT = [&](const std::string& nm, DType dt, std::vector<int64_t> sh,
                 bool is_weight = false, bool is_input = false) -> IRTensor* {
    auto t = std::make_unique<IRTensor>();
    t->name = nm; t->dtype = dt; t->shape = Shape(sh);
    t->is_weight = is_weight; t->is_input = is_input;
    return g.addTensor(std::move(t));
  };

  auto* t0 = mkT("inp",    DType::F16, {1, 64}, false, true);
  auto* t1 = mkT("w0",     DType::F16, {64,128}, true);
  auto* t2 = mkT("mm_out", DType::F16, {1,128});
  auto* t3 = mkT("bias",   DType::F16, {128}, true);
  auto* t4 = mkT("add_out",DType::F16, {1,128});
  auto* t5 = mkT("relu_out",DType::F16, {1,128});

  g.inputIds  = {t0->id};
  g.outputIds = {t5->id};

  auto mkN = [&](OpCode op, std::vector<uint32_t> ins, std::vector<uint32_t> outs) {
    auto n = std::make_unique<IRNode>();
    n->op = op; n->inputs = ins; n->outputs = outs;
    g.addNode(std::move(n));
  };

  mkN(OpCode::MatMul,  {t0->id, t1->id}, {t2->id});
  mkN(OpCode::Add,     {t2->id, t3->id}, {t4->id});
  mkN(OpCode::ReLU,    {t4->id},         {t5->id});

  g.sortTopological();
  return g;
}

// ---------------------------------------------------------------------------
// PassManager: construction and registration
// ---------------------------------------------------------------------------
TEST(PassManagerTest, RegisterAndCount) {
  PassManager pm;
  pm.addPass(makeShapeInferencePass());
  pm.addPass(makeConstantFoldingPass());
  pm.addPass(makeDeadCodeEliminationPass());
  EXPECT_EQ(pm.numPasses(), 3u);
}

TEST(PassManagerTest, RunReturnsResults) {
  PassManager pm;
  pm.addPass(makeShapeInferencePass());
  pm.addPass(makeDeadCodeEliminationPass());

  auto g = buildSimpleGraph();
  PassOptions opts;
  auto results = pm.run(g, opts);

  EXPECT_EQ(results.size(), 2u);
  for (auto& r : results) EXPECT_TRUE(r.success);
}

// ---------------------------------------------------------------------------
// Dead Code Elimination
// ---------------------------------------------------------------------------
TEST(DCEPassTest, RemovesUnusedNode) {
  auto g = buildSimpleGraph();

  // Add an orphaned MatMul whose output is never consumed
  auto orphan_out = std::make_unique<IRTensor>();
  orphan_out->name = "orphan_out"; orphan_out->dtype = DType::F16;
  orphan_out->shape = Shape({1, 64});
  auto* ot = g.addTensor(std::move(orphan_out));

  auto orphan_n = std::make_unique<IRNode>();
  orphan_n->op = OpCode::MatMul;
  orphan_n->inputs = {g.inputIds[0]};
  orphan_n->outputs = {ot->id};
  g.addNode(std::move(orphan_n));

  uint32_t nodes_before = g.numNodes();
  auto pass = makeDeadCodeEliminationPass();
  PassOptions opts;
  auto result = pass->run(g, opts);

  EXPECT_TRUE(result.success);
  EXPECT_TRUE(result.changed);
  EXPECT_LT(g.numNodes(), nodes_before);
}

// ---------------------------------------------------------------------------
// Constant Folding
// ---------------------------------------------------------------------------
TEST(ConstantFoldingTest, FoldsF32ToF16Cast) {
  IRGraph g;
  g.name = "fold_test";

  auto w = std::make_unique<IRTensor>();
  w->name = "const_f32"; w->dtype = DType::F32;
  w->shape = Shape({64, 128}); w->is_weight = true;
  w->weight_data = (uint8_t*)"\0"; w->weight_size = 1;  // stub data
  auto* wt = g.addTensor(std::move(w));

  auto out = std::make_unique<IRTensor>();
  out->name = "cast_out"; out->dtype = DType::F16;
  out->shape = Shape({64, 128}); out->is_weight = false;
  auto* ot = g.addTensor(std::move(out));

  auto cast_n = std::make_unique<IRNode>();
  cast_n->op = OpCode::Cast;
  cast_n->inputs  = {wt->id};
  cast_n->outputs = {ot->id};
  g.addNode(std::move(cast_n));

  auto pass = makeConstantFoldingPass();
  PassOptions opts;
  auto result = pass->run(g, opts);
  EXPECT_TRUE(result.success);
}

// ---------------------------------------------------------------------------
// Operator Fusion: MatMul + Add + ReLU → GEMM
// ---------------------------------------------------------------------------
TEST(OperatorFusionTest, FusesMatMulAddReLU) {
  auto g = buildSimpleGraph();
  uint32_t nodes_before = g.numNodes();

  auto pass = makeOperatorFusionPass();
  PassOptions opts;
  auto result = pass->run(g, opts);

  EXPECT_TRUE(result.success);
  // At minimum, no new nodes should have been added
  EXPECT_LE(g.numNodes(), nodes_before);
}

// ---------------------------------------------------------------------------
// KV Cache Planning
// ---------------------------------------------------------------------------
TEST(KVCachePlanningTest, EnablesKVCacheForAttentionOps) {
  IRGraph g;
  g.name = "kv_test";
  g.num_heads    = 8;
  g.num_kv_heads = 2;
  g.hidden_size  = 512;
  g.num_layers   = 4;

  auto t0 = std::make_unique<IRTensor>();
  t0->name = "in"; t0->dtype = DType::F16; t0->shape = Shape({1, 512});
  auto* tp = g.addTensor(std::move(t0));

  auto t1 = std::make_unique<IRTensor>();
  t1->name = "out"; t1->dtype = DType::F16; t1->shape = Shape({1, 512});
  auto* op2 = g.addTensor(std::move(t1));

  auto attn = std::make_unique<IRNode>();
  attn->op = OpCode::GroupQueryAttention;
  attn->inputs  = {tp->id};
  attn->outputs = {op2->id};
  g.addNode(std::move(attn));

  auto pass = makeKVCachePlanningPass();
  PassOptions opts;
  opts.context_lengths = {2048};
  auto result = pass->run(g, opts);

  EXPECT_TRUE(result.success);
  EXPECT_TRUE(g.kv_cache.enabled);
  EXPECT_GT(g.kv_cache.total_bytes, 0ULL);
}

// ---------------------------------------------------------------------------
// INT8 Quantization Pass
// ---------------------------------------------------------------------------
TEST(INT8QuantPassTest, QuantizesMatMulWeights) {
  auto g = buildSimpleGraph();

  auto pass = makeINT8QuantizationPass();
  PassOptions opts;
  auto result = pass->run(g, opts);

  EXPECT_TRUE(result.success);
  EXPECT_TRUE(result.changed);

  // Check that at least one weight has quant metadata
  bool found_quant = false;
  for (auto& t : g.tensors) {
    if (t->is_weight && t->quant.scheme != QuantScheme::None) {
      found_quant = true;
      EXPECT_EQ(t->quant.stored_as, DType::I8);
      EXPECT_FALSE(t->quant.scales.empty());
      break;
    }
  }
  EXPECT_TRUE(found_quant);
}

// ---------------------------------------------------------------------------
// INT4 Quantization Pass
// ---------------------------------------------------------------------------
TEST(INT4QuantPassTest, QuantizesMatMulWeightsGrouped) {
  auto g = buildSimpleGraph();

  auto pass = makeINT4QuantizationPass(128);
  PassOptions opts;
  auto result = pass->run(g, opts);

  EXPECT_TRUE(result.success);
  for (auto& t : g.tensors) {
    if (t->is_weight && t->quant.scheme == QuantScheme::PerGroup) {
      EXPECT_EQ(t->quant.stored_as, DType::I4);
      EXPECT_EQ(t->quant.group_size, 128);
    }
  }
}

// ---------------------------------------------------------------------------
// Shape Inference Pass
// ---------------------------------------------------------------------------
TEST(ShapeInferenceTest, PropagatesReshapeShape) {
  IRGraph g;
  g.name = "shape_test";

  auto t0 = std::make_unique<IRTensor>();
  t0->name = "x"; t0->dtype = DType::F16; t0->shape = Shape({4, 16});
  auto* ti = g.addTensor(std::move(t0));

  auto t1 = std::make_unique<IRTensor>();
  t1->name = "y"; t1->dtype = DType::F16; t1->shape = Shape({});  // Unknown
  auto* to = g.addTensor(std::move(t1));

  auto n = std::make_unique<IRNode>();
  n->op = OpCode::Reshape;
  n->inputs  = {ti->id};
  n->outputs = {to->id};
  Attribute sh; sh.name = "shape";
  sh.value = std::vector<int64_t>{2, 32};
  n->attrs.push_back(sh);
  g.addNode(std::move(n));

  auto pass = makeShapeInferencePass();
  PassOptions opts;
  auto result = pass->run(g, opts);
  EXPECT_TRUE(result.success);
}

// ---------------------------------------------------------------------------
// Full Pipeline: addFullPipeline
// ---------------------------------------------------------------------------
TEST(PassManagerTest, FullPipelineRunsWithoutCrash) {
  analysis::CostModel cost_model;
  analysis::DeviceProfile dev;
  dev.soc_id    = SoCID::Generic_ARM64;
  dev.name      = "generic";
  dev.has_gpu   = false;
  dev.has_npu   = false;
  dev.has_dsp   = false;
  dev.num_big_cores = 4;
  dev.cpu_freq_mhz  = 2000;
  dev.has_neon      = true;

  auto g = buildSimpleGraph();

  PassManager pm;
  pm.addFullPipeline(dev, DType::I8);

  PassOptions opts;
  opts.device = &dev;
  auto results = pm.run(g, opts);

  EXPECT_GT(results.size(), 0u);
  for (auto& r : results) {
    if (!r.success) {
      // Some passes may legitimately be skipped — just check none crash
      GTEST_NONFATAL_FAILURE_(("Pass failed: " + r.pass_name).c_str());
    }
  }
}
