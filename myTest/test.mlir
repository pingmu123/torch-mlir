module attributes {torch.debug_module_name = "LeNet"} {
  func.func @forward(%arg0: !torch.vtensor<[1,1,28,28],f32>) -> !torch.vtensor<[1,10],f32> {
    %int0 = torch.constant.int 0
    %int1 = torch.constant.int 1
    %float1.000000e00 = torch.constant.float 1.000000e+00
    %int-1 = torch.constant.int -1
    %false = torch.constant.bool false
    %0 = torch.vtensor.literal(dense<[-0.0462377556, 2.318890e-02, 0.0121145593, 0.0342003517, -0.0118353684, -0.0914199352, 0.0956735089, 0.0340309143, 0.0835644081, -0.0435797535]> : tensor<10xf32>) : !torch.vtensor<[10],f32>
    %1 = torch.vtensor.literal(dense_resource<__elided__> : tensor<10x84xf32>) : !torch.vtensor<[10,84],f32>
    %2 = torch.vtensor.literal(dense_resource<__elided__> : tensor<84xf32>) : !torch.vtensor<[84],f32>
    %3 = torch.vtensor.literal(dense_resource<__elided__> : tensor<84x120xf32>) : !torch.vtensor<[84,120],f32>
    %4 = torch.vtensor.literal(dense_resource<__elided__> : tensor<120xf32>) : !torch.vtensor<[120],f32>
    %5 = torch.vtensor.literal(dense_resource<__elided__> : tensor<120x256xf32>) : !torch.vtensor<[120,256],f32>
    %6 = torch.vtensor.literal(dense_resource<__elided__> : tensor<16xf32>) : !torch.vtensor<[16],f32>
    %7 = torch.vtensor.literal(dense_resource<__elided__> : tensor<16x6x5x5xf32>) : !torch.vtensor<[16,6,5,5],f32>
    %8 = torch.vtensor.literal(dense<[0.0428990386, -0.0322512873, -0.168837979, -0.0017533541, -0.0670245886, -0.170828924]> : tensor<6xf32>) : !torch.vtensor<[6],f32>
    %9 = torch.vtensor.literal(dense_resource<__elided__> : tensor<6x1x5x5xf32>) : !torch.vtensor<[6,1,5,5],f32>
    %int2 = torch.constant.int 2
    %10 = torch.prim.ListConstruct %int1, %int1 : (!torch.int, !torch.int) -> !torch.list<int>
    %11 = torch.prim.ListConstruct %int0, %int0 : (!torch.int, !torch.int) -> !torch.list<int>
    %12 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %13 = torch.aten.convolution %arg0, %9, %8, %10, %11, %10, %false, %12, %int1 : !torch.vtensor<[1,1,28,28],f32>, !torch.vtensor<[6,1,5,5],f32>, !torch.vtensor<[6],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,6,24,24],f32>
    %14 = torch.aten.relu %13 : !torch.vtensor<[1,6,24,24],f32> -> !torch.vtensor<[1,6,24,24],f32>
    %15 = torch.prim.ListConstruct %int2, %int2 : (!torch.int, !torch.int) -> !torch.list<int>
    %16 = torch.aten.max_pool2d %14, %15, %15, %11, %10, %false : !torch.vtensor<[1,6,24,24],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,6,12,12],f32>
    %17 = torch.prim.ListConstruct  : () -> !torch.list<int>
    %18 = torch.aten.convolution %16, %7, %6, %10, %11, %10, %false, %17, %int1 : !torch.vtensor<[1,6,12,12],f32>, !torch.vtensor<[16,6,5,5],f32>, !torch.vtensor<[16],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[1,16,8,8],f32>
    %19 = torch.aten.relu %18 : !torch.vtensor<[1,16,8,8],f32> -> !torch.vtensor<[1,16,8,8],f32>
    %20 = torch.aten.max_pool2d %19, %15, %15, %11, %10, %false : !torch.vtensor<[1,16,8,8],f32>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool -> !torch.vtensor<[1,16,4,4],f32>
    %21 = torch.prim.ListConstruct %int1, %int-1 : (!torch.int, !torch.int) -> !torch.list<int>
    %22 = torch.aten.view %20, %21 : !torch.vtensor<[1,16,4,4],f32>, !torch.list<int> -> !torch.vtensor<[1,256],f32>
    %23 = torch.aten.transpose.int %5, %int0, %int1 : !torch.vtensor<[120,256],f32>, !torch.int, !torch.int -> !torch.vtensor<[256,120],f32>
    %24 = torch.aten.mm %22, %23 : !torch.vtensor<[1,256],f32>, !torch.vtensor<[256,120],f32> -> !torch.vtensor<[1,120],f32>
    %25 = torch.aten.add.Tensor %24, %4, %float1.000000e00 : !torch.vtensor<[1,120],f32>, !torch.vtensor<[120],f32>, !torch.float -> !torch.vtensor<[1,120],f32>
    %26 = torch.aten.relu %25 : !torch.vtensor<[1,120],f32> -> !torch.vtensor<[1,120],f32>
    %27 = torch.aten.transpose.int %3, %int0, %int1 : !torch.vtensor<[84,120],f32>, !torch.int, !torch.int -> !torch.vtensor<[120,84],f32>
    %28 = torch.aten.mm %26, %27 : !torch.vtensor<[1,120],f32>, !torch.vtensor<[120,84],f32> -> !torch.vtensor<[1,84],f32>
    %29 = torch.aten.add.Tensor %28, %2, %float1.000000e00 : !torch.vtensor<[1,84],f32>, !torch.vtensor<[84],f32>, !torch.float -> !torch.vtensor<[1,84],f32>
    %30 = torch.aten.relu %29 : !torch.vtensor<[1,84],f32> -> !torch.vtensor<[1,84],f32>
    %31 = torch.aten.transpose.int %1, %int0, %int1 : !torch.vtensor<[10,84],f32>, !torch.int, !torch.int -> !torch.vtensor<[84,10],f32>
    %32 = torch.aten.mm %30, %31 : !torch.vtensor<[1,84],f32>, !torch.vtensor<[84,10],f32> -> !torch.vtensor<[1,10],f32>
    %33 = torch.aten.add.Tensor %32, %0, %float1.000000e00 : !torch.vtensor<[1,10],f32>, !torch.vtensor<[10],f32>, !torch.float -> !torch.vtensor<[1,10],f32>
    return %33 : !torch.vtensor<[1,10],f32>
  }
}