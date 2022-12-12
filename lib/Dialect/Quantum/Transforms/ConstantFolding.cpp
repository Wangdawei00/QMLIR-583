#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
#include "../PassDetail.h"
#include "Dialect/Quantum/QuantumOps.h"
using namespace mlir::quantum;
#include "Dialect/Quantum/Transforms/ConstantFolding.h"

#include <array>
#include <cmath>
using std::sin, std::cos;
using std::atan2;
using std::sqrt;

typedef std::vector<std::vector<double>> RotationMatrix;
class ConstantFoldingPass: public ConstantFoldingPassBase<ConstantFoldingPass>{
 
    void runOnFunction() override;
};


namespace {
#include "Dialect/Quantum/Transforms/ConstantFolding.h.inc"
} // namespace

void ConstantFoldingPass::runOnFunction() {
    OwningRewritePatternList patterns(&getContext());
  populateWithGenerated(patterns);
  if (failed(
          applyPatternsAndFoldGreedily(getFunction(), std::move(patterns)))) {
    signalPassFailure();
  }
}


namespace mlir {

std::unique_ptr<FunctionPass> createConstantFoldingPass() {
  return std::make_unique<ConstantFoldingPass>();
}

} // namespace mlir

static RotationMatrix RotationY(double t) {
  RotationMatrix m = {{cos(t), 0, sin(t)}, {0, 1, 0}, {-sin(t), 0, cos(t)}};
  return m;
}
static RotationMatrix RotationZ(double t) {
  RotationMatrix m = {{cos(t), -sin(t), 0}, {sin(t), cos(t), 0}, {0, 0, 1}};
  return m;
}

RotationMatrix operator*(const RotationMatrix &m1, const RotationMatrix &m2);// {
//   RotationMatrix m = {{0,0,0},{0,0,0},{0,0,0}};
//   for (unsigned i = 0; i < 3; i++) {
//     for (unsigned j = 0; j < 3; j++) {
//       m[i][j] = 0.0;
//       for (unsigned k = 0; k < 3; k++) {
//         m[i][j] += m1[i][k] * m2[k][j];
//       }
//     }
//   }
//   return m;
// }

RotationMatrix YZY2Rotation(double y1, double z1, double y2);// {
//   return RotationY(y2) * RotationZ(z1) * RotationY(y1);
// };

EulerAngles Rotation2ZYZ(const RotationMatrix &m) {
  double z1 = atan2(m[2][1], -m[2][1]);
  double y1 = atan2(sqrt(1 - m[2][2] * m[2][2]), m[2][2]);
  double z2 = atan2(m[1][2], m[0][2]);
  return {z1, y1, z2};
};

EulerAngles EulerAngles::fromYZYtoZYZ(double y1, double z1, double y2) {
  auto m = YZY2Rotation(y1, z1, y2);
  return Rotation2ZYZ(m);
}

Attribute EulerAngles::wrapZYZa1(Builder &builder, Attribute y1, Attribute z11, Attribute z12, Attribute y2) {
  double dy1 = y1.cast<FloatAttr>().getValueAsDouble();
  double dz11 = z11.cast<FloatAttr>().getValueAsDouble();
  double dz12 = z12.cast<FloatAttr>().getValueAsDouble();
  double dy2 = y2.cast<FloatAttr>().getValueAsDouble();
  double dz1 = dz11 + dz12;
  auto ea = fromYZYtoZYZ(dy1, dz1, dy2);
  return builder.getF64FloatAttr(ea.a1);
}

Attribute EulerAngles::wrapZYZa2(Builder &builder, Attribute y1, Attribute z11, Attribute z12, Attribute y2) {
  double dy1 = y1.cast<FloatAttr>().getValueAsDouble();
  double dz11 = z11.cast<FloatAttr>().getValueAsDouble();
  double dz12 = z12.cast<FloatAttr>().getValueAsDouble();
  double dy2 = y2.cast<FloatAttr>().getValueAsDouble();
  double dz1 = dz11 + dz12;
  auto ea = fromYZYtoZYZ(dy1, dz1, dy2);
  return builder.getF64FloatAttr(ea.a2);
}

Attribute EulerAngles::wrapZYZa3(Builder &builder, Attribute y1, Attribute z11, Attribute z12, Attribute y2) {
  double dy1 = y1.cast<FloatAttr>().getValueAsDouble();
  double dz11 = z11.cast<FloatAttr>().getValueAsDouble();
  double dz12 = z12.cast<FloatAttr>().getValueAsDouble();
  double dy2 = y2.cast<FloatAttr>().getValueAsDouble();
  double dz1 = dz11 + dz12;
  auto ea = fromYZYtoZYZ(dy1, dz1, dy2);
  return builder.getF64FloatAttr(ea.a3);
}