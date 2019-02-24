#include <stdint.h>



#ifdef __cplusplus
extern "C" {
#endif
    typedef void* Torch_TensorContext;
    typedef void* Torch_JITModuleContext;
    typedef void* Torch_JITModuleMethodContext;

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

    typedef enum Torch_IValueType {
        Torch_IValueTypeTensor = 1,
        Torch_IValueTypeTuple = 2,
    } Torch_IValueType;

    typedef struct Torch_IValue {
        Torch_IValueType itype;
        void* data_ptr;
    } Torch_IValue;

    typedef struct Torch_IValueTuple {
        Torch_IValue* values;
        size_t length;
    } Torch_IValueTuple;

    typedef struct Torch_ModuleMethodArgument {
        char* name;
        char* typ;
        //Torch_TensorContext default_value;
        //Torch_DataType type;
    } Torch_ModuleMethodArgument;

    void Torch_PrintTensors(Torch_TensorContext* tensors, size_t input_size);

    // Tensor
    Torch_TensorContext Torch_NewTensor(void* data, int64_t* dimensions, int n_dim, Torch_DataType dtype);
    void* Torch_TensorValue(Torch_TensorContext ctx);
    Torch_DataType Torch_TensorType(Torch_TensorContext ctx);
    int64_t* Torch_TensorShape(Torch_TensorContext ctx, size_t* dims);
    void Torch_DeleteTensor(Torch_TensorContext ctx);

    // JIT
    Torch_JITModuleContext Torch_CompileTorchScript(char* script);
    Torch_JITModuleContext Torch_LoadJITModule(char* path);
    void Torch_ExportJITModule(Torch_JITModuleContext ctx, char* path);
    Torch_JITModuleMethodContext Torch_JITModuleGetMethod(Torch_JITModuleContext ctx, char* method);
    Torch_IValue Torch_JITModuleMethodRun(Torch_JITModuleMethodContext ctx, Torch_IValue* inputs, size_t input_size);
    Torch_ModuleMethodArgument* Torch_JITModuleMethodArguments(Torch_JITModuleMethodContext ctx, size_t* res_size);
    Torch_ModuleMethodArgument* Torch_JITModuleMethodReturns(Torch_JITModuleMethodContext ctx, size_t* res_size);
    void Torch_DeleteJITModuleMethod(Torch_JITModuleMethodContext ctx);
    void Torch_DeleteJITModule(Torch_JITModuleContext ctx);


#ifdef __cplusplus
}
#endif