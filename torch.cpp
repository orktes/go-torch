#define _GLIBCXX_USE_CXX11_ABI 0

#include <torch/torch.h>
#include "torch.hpp"
#include <iostream>
#include <stdlib.h>

struct Torch_Tensor {
    torch::Tensor tensor;
};

struct Torch_JITModule {
    std::shared_ptr<torch::jit::script::Module> module;
};

struct Torch_JITModule_Method {
    torch::jit::script::Method& run;
};

torch::TensorOptions Torch_ConvertDataTypeToOptions(Torch_DataType dtype) {
    torch::TensorOptions options;
    switch (dtype) {
        case Torch_Byte:
        options = torch::TensorOptions(torch::kByte);
        break;
        case Torch_Char:
        options = torch::TensorOptions(torch::kChar);
        break;
        case Torch_Short:
        options = torch::TensorOptions(torch::kShort);
        break;
        case Torch_Int:
        options = torch::TensorOptions(torch::kInt);
        break;
        case Torch_Long:
        options = torch::TensorOptions(torch::kLong);
        break;
        case Torch_Half:
        options = torch::TensorOptions(torch::kHalf);
        break;
        case Torch_Float:
        options = torch::TensorOptions(torch::kFloat);
        break;
        case Torch_Double:
        options = torch::TensorOptions(torch::kDouble);
        break;
        default:
        // TODO handle other types
        break;
    }

    return options;
}


Torch_DataType Torch_ConvertScalarTypeToDataType(torch::ScalarType type) {
    Torch_DataType dtype;
    switch (type) {
        case torch::kByte:
        dtype = Torch_Byte;
        break;
        case torch::kChar:
        dtype = Torch_Char;
        break;
        case torch::kShort:
        dtype = Torch_Short;
        break;
        case torch::kInt:
        dtype = Torch_Int;
        break;
        case torch::kLong:
        dtype = Torch_Long;
        break;
        case torch::kHalf:
        dtype = Torch_Half;
        break;
        case torch::kFloat:
        dtype = Torch_Float;
        break;
        case torch::kDouble:
        dtype = Torch_Double;
        break;
        default:
        dtype = Torch_Unknown;
    }

    return dtype;
}

Torch_TensorContext Torch_NewTensor(void* input_data, int64_t* dimensions, int n_dim, Torch_DataType dtype) {
    torch::TensorOptions options = Torch_ConvertDataTypeToOptions(dtype);
    std::vector<int64_t> sizes;
    sizes.assign(dimensions, dimensions + n_dim);

    torch::Tensor ten = torch::from_blob(input_data, torch::IntList(sizes), options);

    auto tensor = new Torch_Tensor();
    tensor->tensor = ten;

    return (void *)tensor;
}

void* Torch_TensorValue(Torch_TensorContext ctx) {
    auto tensor = (Torch_Tensor*)ctx;
    return tensor->tensor.data_ptr();
}

Torch_DataType Torch_TensorType(Torch_TensorContext ctx) {
    auto tensor = (Torch_Tensor*)ctx;
    auto type = tensor->tensor.scalar_type();
    return Torch_ConvertScalarTypeToDataType(type);
}

int64_t* Torch_TensorShape(Torch_TensorContext ctx, size_t* dims){
    auto tensor = (Torch_Tensor*)ctx;
    auto sizes = tensor->tensor.sizes();
    *dims = sizes.size();
    return (int64_t*)sizes.data();
}

void Torch_DeleteTensor(Torch_TensorContext ctx) {
    auto tensor = (Torch_Tensor*)ctx;
    delete tensor;

}

Torch_JITModuleContext Torch_CompileTorchScript(char* cstring_script) {
    std::string script(cstring_script);
    auto mod = new Torch_JITModule();
    mod->module = torch::jit::compile(script);

    return (void *)mod;
}

Torch_JITModuleMethodContext Torch_JITModuleGetMethod(Torch_JITModuleContext ctx, char* cstring_method) {
    std::string method_name(cstring_method);
    auto mod = (Torch_JITModule*)ctx;

    auto met = new Torch_JITModule_Method{
        mod->module->get_method(method_name)
    };


    return (void *)met;
}

void Torch_PrintTensors(Torch_TensorContext* tensors, size_t input_size) {
     for (int i = 0; i < input_size; i++) {
        auto ctx = tensors+i;
        auto tensor = (Torch_Tensor*)*ctx;
        std::cout << tensor->tensor << "\n";
    }
}

Torch_TensorContext* Torch_JITModuleMethodRun(Torch_JITModuleMethodContext ctx, Torch_TensorContext* tensors, size_t input_size, size_t* res_size) {
    auto met = (Torch_JITModule_Method*)ctx;

    std::vector<torch::IValue> inputs;

    for (int i = 0; i < input_size; i++) {
        auto ctx = tensors+i;
        auto tensor = (Torch_Tensor*)*ctx;
        inputs.push_back(tensor->tensor);
    }

    auto res = met->run(inputs);

    if (res.isTensor()) {
        *res_size = 1;
        auto res_ptr = (Torch_TensorContext*)malloc(sizeof(Torch_TensorContext));
        auto tensor = new Torch_Tensor();
        tensor->tensor = res.toTensor();
        *res_ptr = tensor;
        
        return res_ptr;
    }
    
    return NULL;
}