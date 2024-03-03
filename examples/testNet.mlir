module attributes {torch.debug_module_name = "testNet"} {
  func.func @forward(%arg0: !torch.vtensor<[1,1,28,28],f32>) -> !torch.vtensor<[1,2,26,26],f32> {
    %false = torch.constant.bool false
    %0 = torch.vtensor.literal(dense<[-0.20589301, -0.153802127]> : tensor<2xf32>) : !torch.vtensor<[2],f32>
    %1 = torch.vtensor.literal(dense<[[[[0.00372350216, 0.140912771, -0.229578614], [-0.325610042, 0.143054888, -0.232637808], [-0.154405922, 0.198335811, -0.0510406122]]], [[[-0.28384456, -0.310646027, 0.0687524527], [0.221560925, 0.171805114, 0.0790437087], [0.115864083, -0.0922990664, -0.0832136496]]]]> : tensor<2x1x3x3xf32>) : !torch.vtensor<[2,1,3,3],f32>
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %2 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %4 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %5 = torch.aten.convolution %arg0, %1, %0, %2, %3, %2, %false, %4, %int1 : !torch.vtensor<[1,1,28,28],f32>, !torch.vtensor<[2,1,3,3],f32>, !torch.vtensor<[2],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,2,26,26],f32>
    return %5 : !torch.vtensor<[1,2,26,26],f32>
  }
}
