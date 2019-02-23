#include <stdint.h>



#ifdef __cplusplus
extern "C" {
#endif
    typedef enum Torch_DataType {
        Torch_Unknown = 0,
        Torch_Byte = 1,
        Torch_Char = 2,
        Torch_Short = 3,
        Torch_Int = 4,
        Torch_Long = 5,
        Torch_Half = 6,
        Torch_Float = 7,
        Torch_Double = 8,

    } Torch_DataType;

    typedef void* Torch_TensorContext;
    typedef void* Torch_JITModuleContext;
    typedef void* Torch_JITModuleMethodContext;

    void Torch_PrintTensors(Torch_TensorContext* tensors, size_t input_size);

    // Tensor
    Torch_TensorContext Torch_NewTensor(void* data, int64_t* dimensions, int n_dim, Torch_DataType dtype);
    void* Torch_TensorValue(Torch_TensorContext ctx);
    Torch_DataType Torch_TensorType(Torch_TensorContext ctx);
    int64_t* Torch_TensorShape(Torch_TensorContext ctx, size_t* dims);
    void Torch_DeleteTensor(Torch_TensorContext ctx);

    // JIT
    Torch_JITModuleContext Torch_CompileTorchScript(char* script);
    Torch_JITModuleMethodContext Torch_JITModuleGetMethod(Torch_JITModuleContext ctx, char* method);
    Torch_TensorContext* Torch_JITModuleMethodRun(Torch_JITModuleMethodContext ctx, Torch_TensorContext* tensors, size_t input_size, size_t* res_size);

#ifdef __cplusplus
}
#endif