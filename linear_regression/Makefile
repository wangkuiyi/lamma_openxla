SRCS := main.cc bind.cc ../iree_utils/iree_utils.cc
HDRS := bind.h ../iree_utils/iree_utils.h

MLIR := lr.mlir
VMFB := lr-metal.vmfb

all : run

main : ${SRCS} ${HDRS}
	$(info *** Building main ***)
	clang++ -std=c++14 -I.. -o $@ ${SRCS} -F ../../iree-for-apple-platforms/build/runtime/iree.xcframework/macos-arm64_x86_64 -framework iree -framework Metal -framework Foundation

${MLIR} : linear_regression.py export_mlir.py
	$(info *** Export ${MLIR} ***)
	source ../../iree-for-apple-platforms/install.sh; \
	python3 export_mlir.py

${VMFB} : ${MLIR}
	$(info *** Compile ${VMFB} ***)
	source ../../iree-for-apple-platforms/install.sh; \
	iree-compile \
	 --iree-input-type=mhlo \
	 --iree-hal-target-backends=metal \
	 --iree-metal-compile-to-metallib=false \
	 ${MLIR} -o ${VMFB}

run : main ${VMFB}
	$(info *** Run main ${VMFB} ***)
	./main ${VMFB}

clean:
	rm *.vmfb *.mlir main

.PHONY: all clean
