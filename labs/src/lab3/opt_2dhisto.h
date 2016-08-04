#ifndef OPT_KERNEL
#define OPT_KERNEL

void opt_2dhisto(uint32_t* D_input, int, int, uint32_t* g_bins, uint8_t* D_bins);

/* Include below the function headers of any other functions that you implement */

void* AllocateDevice(size_t size);

void MemCpyToDevice(void* dest, void* src, size_t size);

void CopyFromDevice(void* dest, void* src, size_t size);

void FreeDevice(void* addr);

#endif
