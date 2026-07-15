// ============================================================================
// Unit tests for ARM-IR types and graph
// ============================================================================
#include <gtest/gtest.h>
#include "arm_ir/types.h"
#include "arm_ir/ops.h"
#include "arm_ir/graph.h"

using namespace armcc::ir;

// ---------------------------------------------------------------------------
// DType tests
// ---------------------------------------------------------------------------
TEST(DTypeTest, RoundTrip) {
  for (auto dt : {DType::F32, DType::F16, DType::BF16,
                  DType::I8, DType::I4, DType::U8}) {
    EXPECT_EQ(dtypeFromString(dtypeToString(dt)), dt)
        << "Failed for dtype: " << dtypeToString(dt);
  }
}

TEST(DTypeTest, ElementSize) {
  EXPECT_EQ(dtypeElementSize(DType::F32), 4u);
  EXPECT_EQ(dtypeElementSize(DType::F16), 2u);
  EXPECT_EQ(dtypeElementSize(DType::I8),  1u);
  EXPECT_EQ(dtypeElementSize(DType::U8),  1u);
  EXPECT_EQ(dtypeElementSize(DType::I4),  0u);  // sub-byte
}

TEST(DTypeTest, IsFloat) {
  EXPECT_TRUE(dtypeIsFloat(DType::F32));
  EXPECT_TRUE(dtypeIsFloat(DType::F16));
  EXPECT_FALSE(dtypeIsFloat(DType::I8));
  EXPECT_FALSE(dtypeIsFloat(DType::I4));
}

// ---------------------------------------------------------------------------
// Shape tests
// ---------------------------------------------------------------------------
TEST(ShapeTest, NumElements) {
  Shape s({2, 3, 4});
  EXPECT_EQ(s.numElements(), 24);
}

TEST(ShapeTest, DynamicShape) {
  Shape s({-1, 512, 4096});
  EXPECT_TRUE(s.isDynamic());
  EXPECT_EQ(s.numElements(), -1);
}

TEST(ShapeTest, Rank) {
  EXPECT_EQ(Shape({1, 2048}).rank(), 2u);
  EXPECT_EQ(Shape({1, 32, 2048, 128}).rank(), 4u);
}

// ---------------------------------------------------------------------------
// SoCID tests
// ---------------------------------------------------------------------------
TEST(SoCIDTest, RoundTrip) {
  for (auto id : {SoCID::Snapdragon_8_Gen3, SoCID::Dimensity_9300,
                  SoCID::Apple_A18, SoCID::Generic_ARM64}) {
    EXPECT_EQ(socIDFromString(socIDToString(id)), id)
        << "Failed for SoC: " << socIDToString(id);
  }
}

// ---------------------------------------------------------------------------
// OpCode tests
// ---------------------------------------------------------------------------
TEST(OpCodeTest, CommonOps) {
  EXPECT_EQ(opCodeToString(OpCode::MatMul),           "MatMul");
  EXPECT_EQ(opCodeToString(OpCode::GroupQueryAttention),"GroupQueryAttention");
  EXPECT_EQ(opCodeToString(OpCode::RMSNorm),           "RMSNorm");
  EXPECT_EQ(opCodeToString(OpCode::FlashAttention),    "FlashAttention");
}

TEST(OpCodeTest, PredicateHelpers) {
  EXPECT_TRUE(opCodeIsLLMSpecific(OpCode::RopeEmbedding));
  EXPECT_TRUE(opCodeIsLLMSpecific(OpCode::KVCacheRead));
  EXPECT_FALSE(opCodeIsLLMSpecific(OpCode::Add));
  EXPECT_TRUE(opCodeIsHardwareHint(OpCode::HW_Boundary));
  EXPECT_TRUE(opCodeIsFusable(OpCode::ReLU));
}

// ---------------------------------------------------------------------------
// IRGraph construction tests
// ---------------------------------------------------------------------------
TEST(IRGraphTest, AddNodeAndTensor) {
  IRGraph g;
  g.name = "test_graph";

  auto t1 = std::make_unique<IRTensor>();
  t1->name  = "input";
  t1->dtype = DType::F16;
  t1->shape = Shape({1, 512});
  t1->is_input = true;
  auto* tp = g.addTensor(std::move(t1));
  ASSERT_NE(tp, nullptr);
  EXPECT_EQ(tp->id, 1u);

  auto n1 = std::make_unique<IRNode>();
  n1->name = "matmul_0";
  n1->op   = OpCode::MatMul;
  n1->inputs  = {tp->id};
  auto* np = g.addNode(std::move(n1));
  ASSERT_NE(np, nullptr);
  EXPECT_EQ(g.numNodes(), 1u);
}

TEST(IRGraphTest, TopologicalSort) {
  IRGraph g;

  // Build a simple chain: t1 → MatMul → t2 → ReLU → t3
  auto makeT = [&](const std::string& name) -> IRTensor* {
    auto t = std::make_unique<IRTensor>();
    t->name = name; t->dtype = DType::F16; t->shape = Shape({1, 64});
    return g.addTensor(std::move(t));
  };
  auto makeN = [&](OpCode op, uint32_t in, uint32_t out) -> IRNode* {
    auto n = std::make_unique<IRNode>();
    n->op = op; n->inputs = {in}; n->outputs = {out};
    return g.addNode(std::move(n));
  };

  auto* t1 = makeT("t1");
  auto* t2 = makeT("t2");
  auto* t3 = makeT("t3");
  g.inputIds  = {t1->id};
  g.outputIds = {t3->id};

  makeN(OpCode::MatMul, t1->id, t2->id);
  makeN(OpCode::ReLU,   t2->id, t3->id);

  g.sortTopological();
  EXPECT_EQ(g.topoOrder.size(), 2u);
}

TEST(IRGraphTest, FLOPEstimate) {
  IRGraph g;
  // Create a MatMul: [1, 512] x [512, 4096] → [1, 4096]
  auto inp = std::make_unique<IRTensor>();
  inp->dtype = DType::F16; inp->shape = Shape({1, 512}); inp->name = "inp";
  auto out = std::make_unique<IRTensor>();
  out->dtype = DType::F16; out->shape = Shape({1, 4096}); out->name = "out";
  auto* ti = g.addTensor(std::move(inp));
  auto* to = g.addTensor(std::move(out));
  auto n = std::make_unique<IRNode>();
  n->op = OpCode::MatMul; n->inputs = {ti->id}; n->outputs = {to->id};
  g.addNode(std::move(n));

  // FLOPs = 2 * M * K * N = 2 * 1 * 512 * 4096 = 4,194,304
  uint64_t flops = g.totalFLOPs();
  EXPECT_EQ(flops, 2ULL * 1 * 512 * 4096);
}

// ---------------------------------------------------------------------------
// Human-readable IR dump smoke test
// ---------------------------------------------------------------------------
TEST(IRGraphTest, DumpNotEmpty) {
  IRGraph g;
  g.name = "smoke_test";
  g.model_id = "test/model";
  auto t = std::make_unique<IRTensor>();
  t->name = "x"; t->dtype = DType::F16; t->shape = Shape({1, 64});
  t->is_input = true;
  auto* tp = g.addTensor(std::move(t));
  g.inputIds = {tp->id};

  std::string dump = g.dump();
  EXPECT_FALSE(dump.empty());
  EXPECT_NE(dump.find("ARM-IR"), std::string::npos);
  EXPECT_NE(dump.find("smoke_test"), std::string::npos);
}
