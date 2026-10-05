---
name: memory-and-speed
description: Read this file before you measure or change the memory or the speed of the server.
metadata:
  type: project
---

- A model load in glibc keeps its freed conversion buffers in the heap. `memory.model_loading` sets a low `M_MMAP_THRESHOLD` only while a model loads. With a low threshold all the time, torch on the CPU is about 70 % slower. Without it, the verbalizer keeps 1.5 GB instead of 0.5 GB.
- The CPU sessions of ONNX Runtime run without the memory arena. The arena keeps the buffers of the longest chunk, about 300 MB, with no speed gain.
- torch, ONNX Runtime and CTranslate2 see all cores of the host, not the CPU limit of the container. With `--cpus 4` and one thread for each host core, the cold request took 12.1 s instead of 7.1 s. For this reason, `THREADS=0` reads `/sys/fs/cgroup/cpu.max`.
- Read the memory of a process as `VmRSS` in `/proc/<pid>/status`. `wslc stats` and `docker stats` add the page cache.
- Measure a cold load in a new process, with the model files in the page cache. Run each variant three times and take the median. Do not measure while other jobs use the CPU.
- The probe scripts for these measurements are not part of the repository. Write a probe as a script and pipe it into a container on stdin, because the container cannot see the files of the host.
