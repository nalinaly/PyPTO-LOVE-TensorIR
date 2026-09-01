# License and publication status

This repository combines original research code with modified and unmodified
third-party components. It is not legal advice, and the presence of source or
build instructions does not itself grant a license to use or redistribute any
component.

## PyPTO and its runtime

The pinned PyPTO source and runtime are distributed upstream under the **CANN
Open Software License Agreement Version 2.0**. Sections 2.1 and 3.1 limit the
licensed purpose to systems with Huawei AI Processors and/or Software and
restrict development or distribution for systems with other processors.

The NVIDIA SM120 work in this research tree therefore has a direct license
boundary. Personal research, non-commercial intent, an invitation to contact
the author for removal, and public statements about cross-platform compiler
design are not treated here as exceptions to the license. The repository
owner has reviewed this boundary and, for non-profit research purposes,
accepts responsibility for hosting this snapshot on a personal public remote;
this hosting decision does not assert any additional license or exemption,
and rights holders may request removal at any time through GitHub Issues.

Upstream project and license:

- https://github.com/hw-native-sys/pypto
- https://github.com/hw-native-sys/pypto/blob/main/LICENSE

## NVIDIA TensorIR and CUDA Tile

The pinned TensorIR source is distributed under the Apache License 2.0 with
LLVM Exceptions. CUDA Tile carries its own `LICENSE.txt`. Their notices and all
required modification notices must be retained in any authorized source
distribution.

- https://github.com/NVIDIA/tensor-ir
- https://github.com/NVIDIA/cuda-tile

## Project-owned packages

`pypto-framework-plugins` currently carries the Apache License 2.0. The
standalone `pypto-kernels` source snapshot currently has no independent license
file. Until the rights and derivative-work boundary are reviewed, no new public
redistribution license is asserted for the combined release snapshot.

## Interview reference

At approximately 02:34:00 in the linked interview, Dr. Liao Heng describes the
PyPTO design principles and compiler stack as broadly applicable across AI
processors. The documentation may cite or paraphrase that engineering vision,
but must not describe it as a change to the written license or as individual
authorization for this project.

- https://www.bilibili.com/video/BV1nB3u6tERu/?vd_source=f2f41aa7b5e3cc8e0a23942779ccea11

Rights holders may request removal through the repository's GitHub Issues
page. Public hosting of this research snapshot on the author's personal
remote is an owner-acknowledged decision recorded here; it does not modify
any upstream license terms.
