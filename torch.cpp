#define _GLIBCXX_USE_CXX11_ABI 0

#include <torch/torch.h>
#include <torch/script.h>
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

Torch_IValue Torch_ConvertIValueToTorchIValue(torch::IValue value) {
    if (value.isTensor()) {
        auto tensor = new Torch_Tensor();
        tensor->tensor = value.toTensor();
        return Torch_IValue{
            .itype = Torch_IValueTypeTensor,
            .data_ptr = tensor,
        };
    } else if (value.isTuple()) {
        auto elements = value.toTuple()->elements();
        auto tuple = (Torch_IValueTuple*)malloc(sizeof(Torch_IValueTuple));
        auto values = (Torch_IValue*)malloc(sizeof(Torch_IValue) * elements.size());

        for(std::vector<torch::IValue>::size_type i = 0; i != elements.size(); i++) {
            *(values + i) = Torch_ConvertIValueToTorchIValue(elements[i]);
        }

        tuple->values = values;
        tuple->length = elements.size();

        return Torch_IValue{
            .itype = Torch_IValueTypeTuple,
            .data_ptr = tuple,
        };
    }

    return Torch_IValue{};
}

torch::IValue Torch_ConvertTorchIValueToIValue(Torch_IValue value) {
    if (value.itype == Torch_IValueTypeTensor) {
        auto tensor = (Torch_Tensor*)value.data_ptr;
        return tensor->tensor;
    } else if (value.itype == Torch_IValueTypeTuple) {
        auto tuple = (Torch_IValueTuple*)value.data_ptr;
        std::vector<torch::IValue> values;
        values.reserve(tuple->length);

        for (int i = 0; i < tuple->length; i++) {
            auto ival = *(tuple->values+i);
            values.push_back(Torch_ConvertTorchIValueToIValue(ival));
        }

        return torch::jit::Tuple::create(std::move(values));
    }

    // TODO handle this case
    return 0;
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

void Torch_PrintTensors(Torch_TensorContext* tensors, size_t input_size) {
     for (int i = 0; i < input_size; i++) {
        auto ctx = tensors+i;
        auto tensor = (Torch_Tensor*)*ctx;
        std::cout << tensor->tensor << "\n";
    }
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

Torch_JITModuleContext Torch_LoadJITModule(char* cstring_path) {
    std::string module_path(cstring_path);
    auto mod = new Torch_JITModule();
    mod->module = torch::jit::load(module_path);

    return (void *)mod;
}

void Torch_ExportJITModule(Torch_JITModuleContext ctx, char* cstring_path) {
    std::string module_path(cstring_path);
    auto mod = (Torch_JITModule*)ctx;
    mod->module->save(module_path);
}

Torch_JITModuleMethodContext Torch_JITModuleGetMethod(Torch_JITModuleContext ctx, char* cstring_method) {
    std::string method_name(cstring_method);
    auto mod = (Torch_JITModule*)ctx;

    auto met = new Torch_JITModule_Method{
        mod->module->get_method(method_name)
    };

    return (void *)met;
}

Torch_IValue Torch_JITModuleMethodRun(Torch_JITModuleMethodContext ctx, Torch_IValue* inputs, size_t input_size) {
    auto met = (Torch_JITModule_Method*)ctx;

    std::vector<torch::IValue> inputs_vec;

    for (int i = 0; i < input_size; i++) {
        auto ival = *(inputs+i);
        inputs_vec.push_back(Torch_ConvertTorchIValueToIValue(ival));
    }

    auto res = met->run(inputs_vec);
    return Torch_ConvertIValueToTorchIValue(res);
}


Torch_ModuleMethodArgument* Torch_JITModuleMethodArguments(Torch_JITModuleMethodContext ctx, size_t* res_size) {
    auto met = (Torch_JITModule_Method*)ctx;
    auto schema = met->run.getSchema();
    auto arguments = schema.arguments();

    auto result = (Torch_ModuleMethodArgument*)malloc(sizeof(Torch_ModuleMethodArgument)*arguments.size());
    *res_size = arguments.size();

    for(std::vector<torch::Argument>::size_type i = 0; i != arguments.size(); i++) {
        auto name = arguments[i].name();
        char *cstr_name = new char[name.length() + 1];
        strcpy(cstr_name, name.c_str());

        auto type = arguments[i].type()->str();
        char *cstr_type = new char[type.length() + 1];
        strcpy(cstr_type, type.c_str());

        *(result + i) = Torch_ModuleMethodArgument{
            .name = cstr_name,
            .typ = cstr_type,
        };
    }

    return result;
}


Torch_ModuleMethodArgument* Torch_JITModuleMethodReturns(Torch_JITModuleMethodContext ctx, size_t* res_size) {
    auto met = (Torch_JITModule_Method*)ctx;
    auto schema = met->run.getSchema();
    auto arguments = schema.returns();

    auto result = (Torch_ModuleMethodArgument*)malloc(sizeof(Torch_ModuleMethodArgument)*arguments.size());
    *res_size = arguments.size();

    for(std::vector<torch::Argument>::size_type i = 0; i != arguments.size(); i++) {
        auto name = arguments[i].name();
        char *cstr_name = new char[name.length() + 1];
        strcpy(cstr_name, name.c_str());

        auto type = arguments[i].type()->str();
        char *cstr_type = new char[type.length() + 1];
        strcpy(cstr_type, type.c_str());

       
        *(result + i) = Torch_ModuleMethodArgument{
            .name = cstr_name,
            .typ = cstr_type,
        };
    }

    return result;
}


void Torch_DeleteJITModuleMethod(Torch_JITModuleMethodContext ctx) {
    auto med = (Torch_JITModule_Method*)ctx;
    delete med;
}

void Torch_DeleteJITModule(Torch_JITModuleContext ctx) {
    auto mod = (Torch_JITModule*)ctx;
    delete mod;
}
