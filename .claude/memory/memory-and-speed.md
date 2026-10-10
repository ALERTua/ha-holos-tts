---
name: memory-and-speed
description: Read this file before you measure or change the memory or the speed of the server.
metadata:
  type: project
---

- A model load in glibc keeps its freed conversion buffers in the heap. `memory.model_loading` sets a low `M_MMAP_THRESHOLD` only while a model loads. With a low threshold all the time, torch on the CPU is about 70 % slower. Without it, the verbalizer keeps 1.5 GB instead of 0.5 GB.
- The CPU sessions of ONNX Runtime run without the memory arena. The arena keeps the buffers of the longest chunk, about 300 MB, with no speed gain.
- torch, ONNX Runtime and CTranslate2 see all cores of the host, not the CPU limit of the container. With `--cpus 4` and one thread for each host core, the cold request took 12.1 s instead of 7.1 s. For this reason, `THREADS=0` reads `/sys/fs/cgroup/cpu.max`.
- The libraries also ignore the CPU pinning (`--cpuset-cpus`). On an Intel N100 with `--cpuset-cpus=1-3`, the thread defaults of the libraries gave 3.3–8.2 s for each phrase and stalled a VM on the same cores for up to 4.2 s. `THREADS=2` gave 1.55–5.16 s with no stall. For this reason, `THREADS=0` also takes the size of the affinity set when it is smaller than the host.
- Read the memory of a process as `VmRSS` in `/proc/<pid>/status`. `wslc stats` and `docker stats` add the page cache.
- Measure a cold load in a new process, with the model files in the page cache. Run each variant three times and take the median. Do not measure while other jobs use the CPU.
- The probe scripts for these measurements are not part of the repository. Write a probe as a script and pipe it into a container on stdin, because the container cannot see the files of the host.
