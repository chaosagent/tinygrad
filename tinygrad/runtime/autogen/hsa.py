# mypy: ignore-errors
# -*- coding: utf-8 -*-
#
# TARGET arch is: ['-I/opt/rocm/include']
# WORD_SIZE is: 8
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 16
#
import ctypes


def string_cast(char_pointer, encoding='utf-8', errors='strict'):
    value = ctypes.cast(char_pointer, ctypes.c_char_p).value
    if value is not None and encoding is not None:
        value = value.decode(encoding, errors=errors)
    return value


def char_pointer_cast(string, encoding='utf-8'):
    if encoding is not None:
        try:
            string = string.encode(encoding)
        except AttributeError:
            # In Python3, bytes has no encode attribute
            pass
    string = ctypes.c_char_p(string)
    return ctypes.cast(string, ctypes.POINTER(ctypes.c_char))



_libraries = {}
_libraries['libhsa-runtime64.so'] = ctypes.CDLL('/opt/rocm/lib/libhsa-runtime64.so')
class AsDictMixin:
    @classmethod
    def as_dict(cls, self):
        result = {}
        if not isinstance(self, AsDictMixin):
            # not a structure, assume it's already a python object
            return self
        if not hasattr(cls, "_fields_"):
            return result
        # sys.version_info >= (3, 5)
        # for (field, *_) in cls._fields_:  # noqa
        for field_tuple in cls._fields_:  # noqa
            field = field_tuple[0]
            if field.startswith('PADDING_'):
                continue
            value = getattr(self, field)
            type_ = type(value)
            if hasattr(value, "_length_") and hasattr(value, "_type_"):
                # array
                if not hasattr(type_, "as_dict"):
                    value = [v for v in value]
                else:
                    type_ = type_._type_
                    value = [type_.as_dict(v) for v in value]
            elif hasattr(value, "contents") and hasattr(value, "_type_"):
                # pointer
                try:
                    if not hasattr(type_, "as_dict"):
                        value = value.contents
                    else:
                        type_ = type_._type_
                        value = type_.as_dict(value.contents)
                except ValueError:
                    # nullptr
                    value = None
            elif isinstance(value, AsDictMixin):
                # other structure
                value = type_.as_dict(value)
            result[field] = value
        return result


class Structure(ctypes.Structure, AsDictMixin):

    def __init__(self, *args, **kwds):
        # We don't want to use positional arguments fill PADDING_* fields

        args = dict(zip(self.__class__._field_names_(), args))
        args.update(kwds)
        super(Structure, self).__init__(**args)

    @classmethod
    def _field_names_(cls):
        if hasattr(cls, '_fields_'):
            return (f[0] for f in cls._fields_ if not f[0].startswith('PADDING'))
        else:
            return ()

    @classmethod
    def get_type(cls, field):
        for f in cls._fields_:
            if f[0] == field:
                return f[1]
        return None

    @classmethod
    def bind(cls, bound_fields):
        fields = {}
        for name, type_ in cls._fields_:
            if hasattr(type_, "restype"):
                if name in bound_fields:
                    if bound_fields[name] is None:
                        fields[name] = type_()
                    else:
                        # use a closure to capture the callback from the loop scope
                        fields[name] = (
                            type_((lambda callback: lambda *args: callback(*args))(
                                bound_fields[name]))
                        )
                    del bound_fields[name]
                else:
                    # default callback implementation (does nothing)
                    try:
                        default_ = type_(0).restype().value
                    except TypeError:
                        default_ = None
                    fields[name] = type_((
                        lambda default_: lambda *args: default_)(default_))
            else:
                # not a callback function, use default initialization
                if name in bound_fields:
                    fields[name] = bound_fields[name]
                    del bound_fields[name]
                else:
                    fields[name] = type_()
        if len(bound_fields) != 0:
            raise ValueError(
                "Cannot bind the following unknown callback(s) {}.{}".format(
                    cls.__name__, bound_fields.keys()
            ))
        return cls(**fields)


class Union(ctypes.Union, AsDictMixin):
    pass



c_int128 = ctypes.c_ubyte*16
c_uint128 = c_int128
void = None
if ctypes.sizeof(ctypes.c_longdouble) == 16:
    c_long_double_t = ctypes.c_longdouble
else:
    c_long_double_t = ctypes.c_ubyte*16




# values for enumeration 'hsa_status_t'
hsa_status_t__enumvalues = {
    0: 'HSA_STATUS_SUCCESS',
    1: 'HSA_STATUS_INFO_BREAK',
    4096: 'HSA_STATUS_ERROR',
    4097: 'HSA_STATUS_ERROR_INVALID_ARGUMENT',
    4098: 'HSA_STATUS_ERROR_INVALID_QUEUE_CREATION',
    4099: 'HSA_STATUS_ERROR_INVALID_ALLOCATION',
    4100: 'HSA_STATUS_ERROR_INVALID_AGENT',
    4101: 'HSA_STATUS_ERROR_INVALID_REGION',
    4102: 'HSA_STATUS_ERROR_INVALID_SIGNAL',
    4103: 'HSA_STATUS_ERROR_INVALID_QUEUE',
    4104: 'HSA_STATUS_ERROR_OUT_OF_RESOURCES',
    4105: 'HSA_STATUS_ERROR_INVALID_PACKET_FORMAT',
    4106: 'HSA_STATUS_ERROR_RESOURCE_FREE',
    4107: 'HSA_STATUS_ERROR_NOT_INITIALIZED',
    4108: 'HSA_STATUS_ERROR_REFCOUNT_OVERFLOW',
    4109: 'HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS',
    4110: 'HSA_STATUS_ERROR_INVALID_INDEX',
    4111: 'HSA_STATUS_ERROR_INVALID_ISA',
    4119: 'HSA_STATUS_ERROR_INVALID_ISA_NAME',
    4112: 'HSA_STATUS_ERROR_INVALID_CODE_OBJECT',
    4113: 'HSA_STATUS_ERROR_INVALID_EXECUTABLE',
    4114: 'HSA_STATUS_ERROR_FROZEN_EXECUTABLE',
    4115: 'HSA_STATUS_ERROR_INVALID_SYMBOL_NAME',
    4116: 'HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED',
    4117: 'HSA_STATUS_ERROR_VARIABLE_UNDEFINED',
    4118: 'HSA_STATUS_ERROR_EXCEPTION',
    4120: 'HSA_STATUS_ERROR_INVALID_CODE_SYMBOL',
    4121: 'HSA_STATUS_ERROR_INVALID_EXECUTABLE_SYMBOL',
    4128: 'HSA_STATUS_ERROR_INVALID_FILE',
    4129: 'HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER',
    4130: 'HSA_STATUS_ERROR_INVALID_CACHE',
    4131: 'HSA_STATUS_ERROR_INVALID_WAVEFRONT',
    4132: 'HSA_STATUS_ERROR_INVALID_SIGNAL_GROUP',
    4133: 'HSA_STATUS_ERROR_INVALID_RUNTIME_STATE',
    4134: 'HSA_STATUS_ERROR_FATAL',
}
HSA_STATUS_SUCCESS = 0
HSA_STATUS_INFO_BREAK = 1
HSA_STATUS_ERROR = 4096
HSA_STATUS_ERROR_INVALID_ARGUMENT = 4097
HSA_STATUS_ERROR_INVALID_QUEUE_CREATION = 4098
HSA_STATUS_ERROR_INVALID_ALLOCATION = 4099
HSA_STATUS_ERROR_INVALID_AGENT = 4100
HSA_STATUS_ERROR_INVALID_REGION = 4101
HSA_STATUS_ERROR_INVALID_SIGNAL = 4102
HSA_STATUS_ERROR_INVALID_QUEUE = 4103
HSA_STATUS_ERROR_OUT_OF_RESOURCES = 4104
HSA_STATUS_ERROR_INVALID_PACKET_FORMAT = 4105
HSA_STATUS_ERROR_RESOURCE_FREE = 4106
HSA_STATUS_ERROR_NOT_INITIALIZED = 4107
HSA_STATUS_ERROR_REFCOUNT_OVERFLOW = 4108
HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS = 4109
HSA_STATUS_ERROR_INVALID_INDEX = 4110
HSA_STATUS_ERROR_INVALID_ISA = 4111
HSA_STATUS_ERROR_INVALID_ISA_NAME = 4119
HSA_STATUS_ERROR_INVALID_CODE_OBJECT = 4112
HSA_STATUS_ERROR_INVALID_EXECUTABLE = 4113
HSA_STATUS_ERROR_FROZEN_EXECUTABLE = 4114
HSA_STATUS_ERROR_INVALID_SYMBOL_NAME = 4115
HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED = 4116
HSA_STATUS_ERROR_VARIABLE_UNDEFINED = 4117
HSA_STATUS_ERROR_EXCEPTION = 4118
HSA_STATUS_ERROR_INVALID_CODE_SYMBOL = 4120
HSA_STATUS_ERROR_INVALID_EXECUTABLE_SYMBOL = 4121
HSA_STATUS_ERROR_INVALID_FILE = 4128
HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER = 4129
HSA_STATUS_ERROR_INVALID_CACHE = 4130
HSA_STATUS_ERROR_INVALID_WAVEFRONT = 4131
HSA_STATUS_ERROR_INVALID_SIGNAL_GROUP = 4132
HSA_STATUS_ERROR_INVALID_RUNTIME_STATE = 4133
HSA_STATUS_ERROR_FATAL = 4134
hsa_status_t = ctypes.c_uint32 # enum
try:
    hsa_status_string = _libraries['libhsa-runtime64.so'].hsa_status_string
    hsa_status_string.restype = hsa_status_t
    hsa_status_string.argtypes = [hsa_status_t, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass
class struct_hsa_dim3_s(Structure):
    pass

struct_hsa_dim3_s._pack_ = 1 # source:False
struct_hsa_dim3_s._fields_ = [
    ('x', ctypes.c_uint32),
    ('y', ctypes.c_uint32),
    ('z', ctypes.c_uint32),
]

hsa_dim3_t = struct_hsa_dim3_s

# values for enumeration 'hsa_access_permission_t'
hsa_access_permission_t__enumvalues = {
    0: 'HSA_ACCESS_PERMISSION_NONE',
    1: 'HSA_ACCESS_PERMISSION_RO',
    2: 'HSA_ACCESS_PERMISSION_WO',
    3: 'HSA_ACCESS_PERMISSION_RW',
}
HSA_ACCESS_PERMISSION_NONE = 0
HSA_ACCESS_PERMISSION_RO = 1
HSA_ACCESS_PERMISSION_WO = 2
HSA_ACCESS_PERMISSION_RW = 3
hsa_access_permission_t = ctypes.c_uint32 # enum
hsa_file_t = ctypes.c_int32
try:
    hsa_init = _libraries['libhsa-runtime64.so'].hsa_init
    hsa_init.restype = hsa_status_t
    hsa_init.argtypes = []
except AttributeError:
    pass
try:
    hsa_shut_down = _libraries['libhsa-runtime64.so'].hsa_shut_down
    hsa_shut_down.restype = hsa_status_t
    hsa_shut_down.argtypes = []
except AttributeError:
    pass

# values for enumeration 'hsa_endianness_t'
hsa_endianness_t__enumvalues = {
    0: 'HSA_ENDIANNESS_LITTLE',
    1: 'HSA_ENDIANNESS_BIG',
}
HSA_ENDIANNESS_LITTLE = 0
HSA_ENDIANNESS_BIG = 1
hsa_endianness_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_machine_model_t'
hsa_machine_model_t__enumvalues = {
    0: 'HSA_MACHINE_MODEL_SMALL',
    1: 'HSA_MACHINE_MODEL_LARGE',
}
HSA_MACHINE_MODEL_SMALL = 0
HSA_MACHINE_MODEL_LARGE = 1
hsa_machine_model_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_profile_t'
hsa_profile_t__enumvalues = {
    0: 'HSA_PROFILE_BASE',
    1: 'HSA_PROFILE_FULL',
}
HSA_PROFILE_BASE = 0
HSA_PROFILE_FULL = 1
hsa_profile_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_system_info_t'
hsa_system_info_t__enumvalues = {
    0: 'HSA_SYSTEM_INFO_VERSION_MAJOR',
    1: 'HSA_SYSTEM_INFO_VERSION_MINOR',
    2: 'HSA_SYSTEM_INFO_TIMESTAMP',
    3: 'HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY',
    4: 'HSA_SYSTEM_INFO_SIGNAL_MAX_WAIT',
    5: 'HSA_SYSTEM_INFO_ENDIANNESS',
    6: 'HSA_SYSTEM_INFO_MACHINE_MODEL',
    7: 'HSA_SYSTEM_INFO_EXTENSIONS',
    512: 'HSA_AMD_SYSTEM_INFO_BUILD_VERSION',
    513: 'HSA_AMD_SYSTEM_INFO_SVM_SUPPORTED',
    514: 'HSA_AMD_SYSTEM_INFO_SVM_ACCESSIBLE_BY_DEFAULT',
    515: 'HSA_AMD_SYSTEM_INFO_MWAITX_ENABLED',
    516: 'HSA_AMD_SYSTEM_INFO_DMABUF_SUPPORTED',
    517: 'HSA_AMD_SYSTEM_INFO_VIRTUAL_MEM_API_SUPPORTED',
    518: 'HSA_AMD_SYSTEM_INFO_XNACK_ENABLED',
}
HSA_SYSTEM_INFO_VERSION_MAJOR = 0
HSA_SYSTEM_INFO_VERSION_MINOR = 1
HSA_SYSTEM_INFO_TIMESTAMP = 2
HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY = 3
HSA_SYSTEM_INFO_SIGNAL_MAX_WAIT = 4
HSA_SYSTEM_INFO_ENDIANNESS = 5
HSA_SYSTEM_INFO_MACHINE_MODEL = 6
HSA_SYSTEM_INFO_EXTENSIONS = 7
HSA_AMD_SYSTEM_INFO_BUILD_VERSION = 512
HSA_AMD_SYSTEM_INFO_SVM_SUPPORTED = 513
HSA_AMD_SYSTEM_INFO_SVM_ACCESSIBLE_BY_DEFAULT = 514
HSA_AMD_SYSTEM_INFO_MWAITX_ENABLED = 515
HSA_AMD_SYSTEM_INFO_DMABUF_SUPPORTED = 516
HSA_AMD_SYSTEM_INFO_VIRTUAL_MEM_API_SUPPORTED = 517
HSA_AMD_SYSTEM_INFO_XNACK_ENABLED = 518
hsa_system_info_t = ctypes.c_uint32 # enum
try:
    hsa_system_get_info = _libraries['libhsa-runtime64.so'].hsa_system_get_info
    hsa_system_get_info.restype = hsa_status_t
    hsa_system_get_info.argtypes = [hsa_system_info_t, ctypes.POINTER(None)]
except AttributeError:
    pass

# values for enumeration 'hsa_extension_t'
hsa_extension_t__enumvalues = {
    0: 'HSA_EXTENSION_FINALIZER',
    1: 'HSA_EXTENSION_IMAGES',
    2: 'HSA_EXTENSION_PERFORMANCE_COUNTERS',
    3: 'HSA_EXTENSION_PROFILING_EVENTS',
    3: 'HSA_EXTENSION_STD_LAST',
    512: 'HSA_AMD_FIRST_EXTENSION',
    512: 'HSA_EXTENSION_AMD_PROFILER',
    513: 'HSA_EXTENSION_AMD_LOADER',
    514: 'HSA_EXTENSION_AMD_AQLPROFILE',
    514: 'HSA_AMD_LAST_EXTENSION',
}
HSA_EXTENSION_FINALIZER = 0
HSA_EXTENSION_IMAGES = 1
HSA_EXTENSION_PERFORMANCE_COUNTERS = 2
HSA_EXTENSION_PROFILING_EVENTS = 3
HSA_EXTENSION_STD_LAST = 3
HSA_AMD_FIRST_EXTENSION = 512
HSA_EXTENSION_AMD_PROFILER = 512
HSA_EXTENSION_AMD_LOADER = 513
HSA_EXTENSION_AMD_AQLPROFILE = 514
HSA_AMD_LAST_EXTENSION = 514
hsa_extension_t = ctypes.c_uint32 # enum
uint16_t = ctypes.c_uint16
try:
    hsa_extension_get_name = _libraries['libhsa-runtime64.so'].hsa_extension_get_name
    hsa_extension_get_name.restype = hsa_status_t
    hsa_extension_get_name.argtypes = [uint16_t, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError:
    pass
try:
    hsa_system_extension_supported = _libraries['libhsa-runtime64.so'].hsa_system_extension_supported
    hsa_system_extension_supported.restype = hsa_status_t
    hsa_system_extension_supported.argtypes = [uint16_t, uint16_t, uint16_t, ctypes.POINTER(ctypes.c_bool)]
except AttributeError:
    pass
try:
    hsa_system_major_extension_supported = _libraries['libhsa-runtime64.so'].hsa_system_major_extension_supported
    hsa_system_major_extension_supported.restype = hsa_status_t
    hsa_system_major_extension_supported.argtypes = [uint16_t, uint16_t, ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_bool)]
except AttributeError:
    pass
try:
    hsa_system_get_extension_table = _libraries['libhsa-runtime64.so'].hsa_system_get_extension_table
    hsa_system_get_extension_table.restype = hsa_status_t
    hsa_system_get_extension_table.argtypes = [uint16_t, uint16_t, uint16_t, ctypes.POINTER(None)]
except AttributeError:
    pass
size_t = ctypes.c_uint64
try:
    hsa_system_get_major_extension_table = _libraries['libhsa-runtime64.so'].hsa_system_get_major_extension_table
    hsa_system_get_major_extension_table.restype = hsa_status_t
    hsa_system_get_major_extension_table.argtypes = [uint16_t, uint16_t, size_t, ctypes.POINTER(None)]
except AttributeError:
    pass
class struct_hsa_agent_s(Structure):
    pass

struct_hsa_agent_s._pack_ = 1 # source:False
struct_hsa_agent_s._fields_ = [
    ('handle', ctypes.c_uint64),
]

hsa_agent_t = struct_hsa_agent_s

# values for enumeration 'hsa_agent_feature_t'
hsa_agent_feature_t__enumvalues = {
    1: 'HSA_AGENT_FEATURE_KERNEL_DISPATCH',
    2: 'HSA_AGENT_FEATURE_AGENT_DISPATCH',
}
HSA_AGENT_FEATURE_KERNEL_DISPATCH = 1
HSA_AGENT_FEATURE_AGENT_DISPATCH = 2
hsa_agent_feature_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_device_type_t'
hsa_device_type_t__enumvalues = {
    0: 'HSA_DEVICE_TYPE_CPU',
    1: 'HSA_DEVICE_TYPE_GPU',
    2: 'HSA_DEVICE_TYPE_DSP',
}
HSA_DEVICE_TYPE_CPU = 0
HSA_DEVICE_TYPE_GPU = 1
HSA_DEVICE_TYPE_DSP = 2
hsa_device_type_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_default_float_rounding_mode_t'
hsa_default_float_rounding_mode_t__enumvalues = {
    0: 'HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT',
    1: 'HSA_DEFAULT_FLOAT_ROUNDING_MODE_ZERO',
    2: 'HSA_DEFAULT_FLOAT_ROUNDING_MODE_NEAR',
}
HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT = 0
HSA_DEFAULT_FLOAT_ROUNDING_MODE_ZERO = 1
HSA_DEFAULT_FLOAT_ROUNDING_MODE_NEAR = 2
hsa_default_float_rounding_mode_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_agent_info_t'
hsa_agent_info_t__enumvalues = {
    0: 'HSA_AGENT_INFO_NAME',
    1: 'HSA_AGENT_INFO_VENDOR_NAME',
    2: 'HSA_AGENT_INFO_FEATURE',
    3: 'HSA_AGENT_INFO_MACHINE_MODEL',
    4: 'HSA_AGENT_INFO_PROFILE',
    5: 'HSA_AGENT_INFO_DEFAULT_FLOAT_ROUNDING_MODE',
    23: 'HSA_AGENT_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES',
    24: 'HSA_AGENT_INFO_FAST_F16_OPERATION',
    6: 'HSA_AGENT_INFO_WAVEFRONT_SIZE',
    7: 'HSA_AGENT_INFO_WORKGROUP_MAX_DIM',
    8: 'HSA_AGENT_INFO_WORKGROUP_MAX_SIZE',
    9: 'HSA_AGENT_INFO_GRID_MAX_DIM',
    10: 'HSA_AGENT_INFO_GRID_MAX_SIZE',
    11: 'HSA_AGENT_INFO_FBARRIER_MAX_SIZE',
    12: 'HSA_AGENT_INFO_QUEUES_MAX',
    13: 'HSA_AGENT_INFO_QUEUE_MIN_SIZE',
    14: 'HSA_AGENT_INFO_QUEUE_MAX_SIZE',
    15: 'HSA_AGENT_INFO_QUEUE_TYPE',
    16: 'HSA_AGENT_INFO_NODE',
    17: 'HSA_AGENT_INFO_DEVICE',
    18: 'HSA_AGENT_INFO_CACHE_SIZE',
    19: 'HSA_AGENT_INFO_ISA',
    20: 'HSA_AGENT_INFO_EXTENSIONS',
    21: 'HSA_AGENT_INFO_VERSION_MAJOR',
    22: 'HSA_AGENT_INFO_VERSION_MINOR',
    2147483647: 'HSA_AGENT_INFO_LAST',
}
HSA_AGENT_INFO_NAME = 0
HSA_AGENT_INFO_VENDOR_NAME = 1
HSA_AGENT_INFO_FEATURE = 2
HSA_AGENT_INFO_MACHINE_MODEL = 3
HSA_AGENT_INFO_PROFILE = 4
HSA_AGENT_INFO_DEFAULT_FLOAT_ROUNDING_MODE = 5
HSA_AGENT_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES = 23
HSA_AGENT_INFO_FAST_F16_OPERATION = 24
HSA_AGENT_INFO_WAVEFRONT_SIZE = 6
HSA_AGENT_INFO_WORKGROUP_MAX_DIM = 7
HSA_AGENT_INFO_WORKGROUP_MAX_SIZE = 8
HSA_AGENT_INFO_GRID_MAX_DIM = 9
HSA_AGENT_INFO_GRID_MAX_SIZE = 10
HSA_AGENT_INFO_FBARRIER_MAX_SIZE = 11
HSA_AGENT_INFO_QUEUES_MAX = 12
HSA_AGENT_INFO_QUEUE_MIN_SIZE = 13
HSA_AGENT_INFO_QUEUE_MAX_SIZE = 14
HSA_AGENT_INFO_QUEUE_TYPE = 15
HSA_AGENT_INFO_NODE = 16
HSA_AGENT_INFO_DEVICE = 17
HSA_AGENT_INFO_CACHE_SIZE = 18
HSA_AGENT_INFO_ISA = 19
HSA_AGENT_INFO_EXTENSIONS = 20
HSA_AGENT_INFO_VERSION_MAJOR = 21
HSA_AGENT_INFO_VERSION_MINOR = 22
HSA_AGENT_INFO_LAST = 2147483647
hsa_agent_info_t = ctypes.c_uint32 # enum
try:
    hsa_agent_get_info = _libraries['libhsa-runtime64.so'].hsa_agent_get_info
    hsa_agent_get_info.restype = hsa_status_t
    hsa_agent_get_info.argtypes = [hsa_agent_t, hsa_agent_info_t, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_iterate_agents = _libraries['libhsa-runtime64.so'].hsa_iterate_agents
    hsa_iterate_agents.restype = hsa_status_t
    hsa_iterate_agents.argtypes = [ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_agent_s, ctypes.POINTER(None)), ctypes.POINTER(None)]
except AttributeError:
    pass

# values for enumeration 'hsa_exception_policy_t'
hsa_exception_policy_t__enumvalues = {
    1: 'HSA_EXCEPTION_POLICY_BREAK',
    2: 'HSA_EXCEPTION_POLICY_DETECT',
}
HSA_EXCEPTION_POLICY_BREAK = 1
HSA_EXCEPTION_POLICY_DETECT = 2
hsa_exception_policy_t = ctypes.c_uint32 # enum
try:
    hsa_agent_get_exception_policies = _libraries['libhsa-runtime64.so'].hsa_agent_get_exception_policies
    hsa_agent_get_exception_policies.restype = hsa_status_t
    hsa_agent_get_exception_policies.argtypes = [hsa_agent_t, hsa_profile_t, ctypes.POINTER(ctypes.c_uint16)]
except AttributeError:
    pass
class struct_hsa_cache_s(Structure):
    pass

struct_hsa_cache_s._pack_ = 1 # source:False
struct_hsa_cache_s._fields_ = [
    ('handle', ctypes.c_uint64),
]

hsa_cache_t = struct_hsa_cache_s

# values for enumeration 'hsa_cache_info_t'
hsa_cache_info_t__enumvalues = {
    0: 'HSA_CACHE_INFO_NAME_LENGTH',
    1: 'HSA_CACHE_INFO_NAME',
    2: 'HSA_CACHE_INFO_LEVEL',
    3: 'HSA_CACHE_INFO_SIZE',
}
HSA_CACHE_INFO_NAME_LENGTH = 0
HSA_CACHE_INFO_NAME = 1
HSA_CACHE_INFO_LEVEL = 2
HSA_CACHE_INFO_SIZE = 3
hsa_cache_info_t = ctypes.c_uint32 # enum
try:
    hsa_cache_get_info = _libraries['libhsa-runtime64.so'].hsa_cache_get_info
    hsa_cache_get_info.restype = hsa_status_t
    hsa_cache_get_info.argtypes = [hsa_cache_t, hsa_cache_info_t, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_agent_iterate_caches = _libraries['libhsa-runtime64.so'].hsa_agent_iterate_caches
    hsa_agent_iterate_caches.restype = hsa_status_t
    hsa_agent_iterate_caches.argtypes = [hsa_agent_t, ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_cache_s, ctypes.POINTER(None)), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_agent_extension_supported = _libraries['libhsa-runtime64.so'].hsa_agent_extension_supported
    hsa_agent_extension_supported.restype = hsa_status_t
    hsa_agent_extension_supported.argtypes = [uint16_t, hsa_agent_t, uint16_t, uint16_t, ctypes.POINTER(ctypes.c_bool)]
except AttributeError:
    pass
try:
    hsa_agent_major_extension_supported = _libraries['libhsa-runtime64.so'].hsa_agent_major_extension_supported
    hsa_agent_major_extension_supported.restype = hsa_status_t
    hsa_agent_major_extension_supported.argtypes = [uint16_t, hsa_agent_t, uint16_t, ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_bool)]
except AttributeError:
    pass
class struct_hsa_signal_s(Structure):
    pass

struct_hsa_signal_s._pack_ = 1 # source:False
struct_hsa_signal_s._fields_ = [
    ('handle', ctypes.c_uint64),
]

hsa_signal_t = struct_hsa_signal_s
hsa_signal_value_t = ctypes.c_int64
uint32_t = ctypes.c_uint32
try:
    hsa_signal_create = _libraries['libhsa-runtime64.so'].hsa_signal_create
    hsa_signal_create.restype = hsa_status_t
    hsa_signal_create.argtypes = [hsa_signal_value_t, uint32_t, ctypes.POINTER(struct_hsa_agent_s), ctypes.POINTER(struct_hsa_signal_s)]
except AttributeError:
    pass
try:
    hsa_signal_destroy = _libraries['libhsa-runtime64.so'].hsa_signal_destroy
    hsa_signal_destroy.restype = hsa_status_t
    hsa_signal_destroy.argtypes = [hsa_signal_t]
except AttributeError:
    pass
try:
    hsa_signal_load_scacquire = _libraries['libhsa-runtime64.so'].hsa_signal_load_scacquire
    hsa_signal_load_scacquire.restype = hsa_signal_value_t
    hsa_signal_load_scacquire.argtypes = [hsa_signal_t]
except AttributeError:
    pass
try:
    hsa_signal_load_relaxed = _libraries['libhsa-runtime64.so'].hsa_signal_load_relaxed
    hsa_signal_load_relaxed.restype = hsa_signal_value_t
    hsa_signal_load_relaxed.argtypes = [hsa_signal_t]
except AttributeError:
    pass
try:
    hsa_signal_load_acquire = _libraries['libhsa-runtime64.so'].hsa_signal_load_acquire
    hsa_signal_load_acquire.restype = hsa_signal_value_t
    hsa_signal_load_acquire.argtypes = [hsa_signal_t]
except AttributeError:
    pass
try:
    hsa_signal_store_relaxed = _libraries['libhsa-runtime64.so'].hsa_signal_store_relaxed
    hsa_signal_store_relaxed.restype = None
    hsa_signal_store_relaxed.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_store_screlease = _libraries['libhsa-runtime64.so'].hsa_signal_store_screlease
    hsa_signal_store_screlease.restype = None
    hsa_signal_store_screlease.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_store_release = _libraries['libhsa-runtime64.so'].hsa_signal_store_release
    hsa_signal_store_release.restype = None
    hsa_signal_store_release.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_silent_store_relaxed = _libraries['libhsa-runtime64.so'].hsa_signal_silent_store_relaxed
    hsa_signal_silent_store_relaxed.restype = None
    hsa_signal_silent_store_relaxed.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_silent_store_screlease = _libraries['libhsa-runtime64.so'].hsa_signal_silent_store_screlease
    hsa_signal_silent_store_screlease.restype = None
    hsa_signal_silent_store_screlease.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_exchange_scacq_screl = _libraries['libhsa-runtime64.so'].hsa_signal_exchange_scacq_screl
    hsa_signal_exchange_scacq_screl.restype = hsa_signal_value_t
    hsa_signal_exchange_scacq_screl.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_exchange_acq_rel = _libraries['libhsa-runtime64.so'].hsa_signal_exchange_acq_rel
    hsa_signal_exchange_acq_rel.restype = hsa_signal_value_t
    hsa_signal_exchange_acq_rel.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_exchange_scacquire = _libraries['libhsa-runtime64.so'].hsa_signal_exchange_scacquire
    hsa_signal_exchange_scacquire.restype = hsa_signal_value_t
    hsa_signal_exchange_scacquire.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_exchange_acquire = _libraries['libhsa-runtime64.so'].hsa_signal_exchange_acquire
    hsa_signal_exchange_acquire.restype = hsa_signal_value_t
    hsa_signal_exchange_acquire.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_exchange_relaxed = _libraries['libhsa-runtime64.so'].hsa_signal_exchange_relaxed
    hsa_signal_exchange_relaxed.restype = hsa_signal_value_t
    hsa_signal_exchange_relaxed.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_exchange_screlease = _libraries['libhsa-runtime64.so'].hsa_signal_exchange_screlease
    hsa_signal_exchange_screlease.restype = hsa_signal_value_t
    hsa_signal_exchange_screlease.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_exchange_release = _libraries['libhsa-runtime64.so'].hsa_signal_exchange_release
    hsa_signal_exchange_release.restype = hsa_signal_value_t
    hsa_signal_exchange_release.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_cas_scacq_screl = _libraries['libhsa-runtime64.so'].hsa_signal_cas_scacq_screl
    hsa_signal_cas_scacq_screl.restype = hsa_signal_value_t
    hsa_signal_cas_scacq_screl.argtypes = [hsa_signal_t, hsa_signal_value_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_cas_acq_rel = _libraries['libhsa-runtime64.so'].hsa_signal_cas_acq_rel
    hsa_signal_cas_acq_rel.restype = hsa_signal_value_t
    hsa_signal_cas_acq_rel.argtypes = [hsa_signal_t, hsa_signal_value_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_cas_scacquire = _libraries['libhsa-runtime64.so'].hsa_signal_cas_scacquire
    hsa_signal_cas_scacquire.restype = hsa_signal_value_t
    hsa_signal_cas_scacquire.argtypes = [hsa_signal_t, hsa_signal_value_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_cas_acquire = _libraries['libhsa-runtime64.so'].hsa_signal_cas_acquire
    hsa_signal_cas_acquire.restype = hsa_signal_value_t
    hsa_signal_cas_acquire.argtypes = [hsa_signal_t, hsa_signal_value_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_cas_relaxed = _libraries['libhsa-runtime64.so'].hsa_signal_cas_relaxed
    hsa_signal_cas_relaxed.restype = hsa_signal_value_t
    hsa_signal_cas_relaxed.argtypes = [hsa_signal_t, hsa_signal_value_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_cas_screlease = _libraries['libhsa-runtime64.so'].hsa_signal_cas_screlease
    hsa_signal_cas_screlease.restype = hsa_signal_value_t
    hsa_signal_cas_screlease.argtypes = [hsa_signal_t, hsa_signal_value_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_cas_release = _libraries['libhsa-runtime64.so'].hsa_signal_cas_release
    hsa_signal_cas_release.restype = hsa_signal_value_t
    hsa_signal_cas_release.argtypes = [hsa_signal_t, hsa_signal_value_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_add_scacq_screl = _libraries['libhsa-runtime64.so'].hsa_signal_add_scacq_screl
    hsa_signal_add_scacq_screl.restype = None
    hsa_signal_add_scacq_screl.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_add_acq_rel = _libraries['libhsa-runtime64.so'].hsa_signal_add_acq_rel
    hsa_signal_add_acq_rel.restype = None
    hsa_signal_add_acq_rel.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_add_scacquire = _libraries['libhsa-runtime64.so'].hsa_signal_add_scacquire
    hsa_signal_add_scacquire.restype = None
    hsa_signal_add_scacquire.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_add_acquire = _libraries['libhsa-runtime64.so'].hsa_signal_add_acquire
    hsa_signal_add_acquire.restype = None
    hsa_signal_add_acquire.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_add_relaxed = _libraries['libhsa-runtime64.so'].hsa_signal_add_relaxed
    hsa_signal_add_relaxed.restype = None
    hsa_signal_add_relaxed.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_add_screlease = _libraries['libhsa-runtime64.so'].hsa_signal_add_screlease
    hsa_signal_add_screlease.restype = None
    hsa_signal_add_screlease.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_add_release = _libraries['libhsa-runtime64.so'].hsa_signal_add_release
    hsa_signal_add_release.restype = None
    hsa_signal_add_release.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_subtract_scacq_screl = _libraries['libhsa-runtime64.so'].hsa_signal_subtract_scacq_screl
    hsa_signal_subtract_scacq_screl.restype = None
    hsa_signal_subtract_scacq_screl.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_subtract_acq_rel = _libraries['libhsa-runtime64.so'].hsa_signal_subtract_acq_rel
    hsa_signal_subtract_acq_rel.restype = None
    hsa_signal_subtract_acq_rel.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_subtract_scacquire = _libraries['libhsa-runtime64.so'].hsa_signal_subtract_scacquire
    hsa_signal_subtract_scacquire.restype = None
    hsa_signal_subtract_scacquire.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_subtract_acquire = _libraries['libhsa-runtime64.so'].hsa_signal_subtract_acquire
    hsa_signal_subtract_acquire.restype = None
    hsa_signal_subtract_acquire.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_subtract_relaxed = _libraries['libhsa-runtime64.so'].hsa_signal_subtract_relaxed
    hsa_signal_subtract_relaxed.restype = None
    hsa_signal_subtract_relaxed.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_subtract_screlease = _libraries['libhsa-runtime64.so'].hsa_signal_subtract_screlease
    hsa_signal_subtract_screlease.restype = None
    hsa_signal_subtract_screlease.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_subtract_release = _libraries['libhsa-runtime64.so'].hsa_signal_subtract_release
    hsa_signal_subtract_release.restype = None
    hsa_signal_subtract_release.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_and_scacq_screl = _libraries['libhsa-runtime64.so'].hsa_signal_and_scacq_screl
    hsa_signal_and_scacq_screl.restype = None
    hsa_signal_and_scacq_screl.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_and_acq_rel = _libraries['libhsa-runtime64.so'].hsa_signal_and_acq_rel
    hsa_signal_and_acq_rel.restype = None
    hsa_signal_and_acq_rel.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_and_scacquire = _libraries['libhsa-runtime64.so'].hsa_signal_and_scacquire
    hsa_signal_and_scacquire.restype = None
    hsa_signal_and_scacquire.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_and_acquire = _libraries['libhsa-runtime64.so'].hsa_signal_and_acquire
    hsa_signal_and_acquire.restype = None
    hsa_signal_and_acquire.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_and_relaxed = _libraries['libhsa-runtime64.so'].hsa_signal_and_relaxed
    hsa_signal_and_relaxed.restype = None
    hsa_signal_and_relaxed.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_and_screlease = _libraries['libhsa-runtime64.so'].hsa_signal_and_screlease
    hsa_signal_and_screlease.restype = None
    hsa_signal_and_screlease.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_and_release = _libraries['libhsa-runtime64.so'].hsa_signal_and_release
    hsa_signal_and_release.restype = None
    hsa_signal_and_release.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_or_scacq_screl = _libraries['libhsa-runtime64.so'].hsa_signal_or_scacq_screl
    hsa_signal_or_scacq_screl.restype = None
    hsa_signal_or_scacq_screl.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_or_acq_rel = _libraries['libhsa-runtime64.so'].hsa_signal_or_acq_rel
    hsa_signal_or_acq_rel.restype = None
    hsa_signal_or_acq_rel.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_or_scacquire = _libraries['libhsa-runtime64.so'].hsa_signal_or_scacquire
    hsa_signal_or_scacquire.restype = None
    hsa_signal_or_scacquire.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_or_acquire = _libraries['libhsa-runtime64.so'].hsa_signal_or_acquire
    hsa_signal_or_acquire.restype = None
    hsa_signal_or_acquire.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_or_relaxed = _libraries['libhsa-runtime64.so'].hsa_signal_or_relaxed
    hsa_signal_or_relaxed.restype = None
    hsa_signal_or_relaxed.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_or_screlease = _libraries['libhsa-runtime64.so'].hsa_signal_or_screlease
    hsa_signal_or_screlease.restype = None
    hsa_signal_or_screlease.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_or_release = _libraries['libhsa-runtime64.so'].hsa_signal_or_release
    hsa_signal_or_release.restype = None
    hsa_signal_or_release.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_xor_scacq_screl = _libraries['libhsa-runtime64.so'].hsa_signal_xor_scacq_screl
    hsa_signal_xor_scacq_screl.restype = None
    hsa_signal_xor_scacq_screl.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_xor_acq_rel = _libraries['libhsa-runtime64.so'].hsa_signal_xor_acq_rel
    hsa_signal_xor_acq_rel.restype = None
    hsa_signal_xor_acq_rel.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_xor_scacquire = _libraries['libhsa-runtime64.so'].hsa_signal_xor_scacquire
    hsa_signal_xor_scacquire.restype = None
    hsa_signal_xor_scacquire.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_xor_acquire = _libraries['libhsa-runtime64.so'].hsa_signal_xor_acquire
    hsa_signal_xor_acquire.restype = None
    hsa_signal_xor_acquire.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_xor_relaxed = _libraries['libhsa-runtime64.so'].hsa_signal_xor_relaxed
    hsa_signal_xor_relaxed.restype = None
    hsa_signal_xor_relaxed.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_xor_screlease = _libraries['libhsa-runtime64.so'].hsa_signal_xor_screlease
    hsa_signal_xor_screlease.restype = None
    hsa_signal_xor_screlease.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass
try:
    hsa_signal_xor_release = _libraries['libhsa-runtime64.so'].hsa_signal_xor_release
    hsa_signal_xor_release.restype = None
    hsa_signal_xor_release.argtypes = [hsa_signal_t, hsa_signal_value_t]
except AttributeError:
    pass

# values for enumeration 'hsa_signal_condition_t'
hsa_signal_condition_t__enumvalues = {
    0: 'HSA_SIGNAL_CONDITION_EQ',
    1: 'HSA_SIGNAL_CONDITION_NE',
    2: 'HSA_SIGNAL_CONDITION_LT',
    3: 'HSA_SIGNAL_CONDITION_GTE',
}
HSA_SIGNAL_CONDITION_EQ = 0
HSA_SIGNAL_CONDITION_NE = 1
HSA_SIGNAL_CONDITION_LT = 2
HSA_SIGNAL_CONDITION_GTE = 3
hsa_signal_condition_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_wait_state_t'
hsa_wait_state_t__enumvalues = {
    0: 'HSA_WAIT_STATE_BLOCKED',
    1: 'HSA_WAIT_STATE_ACTIVE',
}
HSA_WAIT_STATE_BLOCKED = 0
HSA_WAIT_STATE_ACTIVE = 1
hsa_wait_state_t = ctypes.c_uint32 # enum
uint64_t = ctypes.c_uint64
try:
    hsa_signal_wait_scacquire = _libraries['libhsa-runtime64.so'].hsa_signal_wait_scacquire
    hsa_signal_wait_scacquire.restype = hsa_signal_value_t
    hsa_signal_wait_scacquire.argtypes = [hsa_signal_t, hsa_signal_condition_t, hsa_signal_value_t, uint64_t, hsa_wait_state_t]
except AttributeError:
    pass
try:
    hsa_signal_wait_relaxed = _libraries['libhsa-runtime64.so'].hsa_signal_wait_relaxed
    hsa_signal_wait_relaxed.restype = hsa_signal_value_t
    hsa_signal_wait_relaxed.argtypes = [hsa_signal_t, hsa_signal_condition_t, hsa_signal_value_t, uint64_t, hsa_wait_state_t]
except AttributeError:
    pass
try:
    hsa_signal_wait_acquire = _libraries['libhsa-runtime64.so'].hsa_signal_wait_acquire
    hsa_signal_wait_acquire.restype = hsa_signal_value_t
    hsa_signal_wait_acquire.argtypes = [hsa_signal_t, hsa_signal_condition_t, hsa_signal_value_t, uint64_t, hsa_wait_state_t]
except AttributeError:
    pass
class struct_hsa_signal_group_s(Structure):
    pass

struct_hsa_signal_group_s._pack_ = 1 # source:False
struct_hsa_signal_group_s._fields_ = [
    ('handle', ctypes.c_uint64),
]

hsa_signal_group_t = struct_hsa_signal_group_s
try:
    hsa_signal_group_create = _libraries['libhsa-runtime64.so'].hsa_signal_group_create
    hsa_signal_group_create.restype = hsa_status_t
    hsa_signal_group_create.argtypes = [uint32_t, ctypes.POINTER(struct_hsa_signal_s), uint32_t, ctypes.POINTER(struct_hsa_agent_s), ctypes.POINTER(struct_hsa_signal_group_s)]
except AttributeError:
    pass
try:
    hsa_signal_group_destroy = _libraries['libhsa-runtime64.so'].hsa_signal_group_destroy
    hsa_signal_group_destroy.restype = hsa_status_t
    hsa_signal_group_destroy.argtypes = [hsa_signal_group_t]
except AttributeError:
    pass
try:
    hsa_signal_group_wait_any_scacquire = _libraries['libhsa-runtime64.so'].hsa_signal_group_wait_any_scacquire
    hsa_signal_group_wait_any_scacquire.restype = hsa_status_t
    hsa_signal_group_wait_any_scacquire.argtypes = [hsa_signal_group_t, ctypes.POINTER(hsa_signal_condition_t), ctypes.POINTER(ctypes.c_int64), hsa_wait_state_t, ctypes.POINTER(struct_hsa_signal_s), ctypes.POINTER(ctypes.c_int64)]
except AttributeError:
    pass
try:
    hsa_signal_group_wait_any_relaxed = _libraries['libhsa-runtime64.so'].hsa_signal_group_wait_any_relaxed
    hsa_signal_group_wait_any_relaxed.restype = hsa_status_t
    hsa_signal_group_wait_any_relaxed.argtypes = [hsa_signal_group_t, ctypes.POINTER(hsa_signal_condition_t), ctypes.POINTER(ctypes.c_int64), hsa_wait_state_t, ctypes.POINTER(struct_hsa_signal_s), ctypes.POINTER(ctypes.c_int64)]
except AttributeError:
    pass
class struct_hsa_region_s(Structure):
    pass

struct_hsa_region_s._pack_ = 1 # source:False
struct_hsa_region_s._fields_ = [
    ('handle', ctypes.c_uint64),
]

hsa_region_t = struct_hsa_region_s

# values for enumeration 'hsa_queue_type_t'
hsa_queue_type_t__enumvalues = {
    0: 'HSA_QUEUE_TYPE_MULTI',
    1: 'HSA_QUEUE_TYPE_SINGLE',
    2: 'HSA_QUEUE_TYPE_COOPERATIVE',
}
HSA_QUEUE_TYPE_MULTI = 0
HSA_QUEUE_TYPE_SINGLE = 1
HSA_QUEUE_TYPE_COOPERATIVE = 2
hsa_queue_type_t = ctypes.c_uint32 # enum
hsa_queue_type32_t = ctypes.c_uint32

# values for enumeration 'hsa_queue_feature_t'
hsa_queue_feature_t__enumvalues = {
    1: 'HSA_QUEUE_FEATURE_KERNEL_DISPATCH',
    2: 'HSA_QUEUE_FEATURE_AGENT_DISPATCH',
}
HSA_QUEUE_FEATURE_KERNEL_DISPATCH = 1
HSA_QUEUE_FEATURE_AGENT_DISPATCH = 2
hsa_queue_feature_t = ctypes.c_uint32 # enum
class struct_hsa_queue_s(Structure):
    pass

struct_hsa_queue_s._pack_ = 1 # source:False
struct_hsa_queue_s._fields_ = [
    ('type', ctypes.c_uint32),
    ('features', ctypes.c_uint32),
    ('base_address', ctypes.POINTER(None)),
    ('doorbell_signal', hsa_signal_t),
    ('size', ctypes.c_uint32),
    ('reserved1', ctypes.c_uint32),
    ('id', ctypes.c_uint64),
]

hsa_queue_t = struct_hsa_queue_s
try:
    hsa_queue_create = _libraries['libhsa-runtime64.so'].hsa_queue_create
    hsa_queue_create.restype = hsa_status_t
    hsa_queue_create.argtypes = [hsa_agent_t, uint32_t, hsa_queue_type32_t, ctypes.CFUNCTYPE(None, hsa_status_t, ctypes.POINTER(struct_hsa_queue_s), ctypes.POINTER(None)), ctypes.POINTER(None), uint32_t, uint32_t, ctypes.POINTER(ctypes.POINTER(struct_hsa_queue_s))]
except AttributeError:
    pass
try:
    hsa_soft_queue_create = _libraries['libhsa-runtime64.so'].hsa_soft_queue_create
    hsa_soft_queue_create.restype = hsa_status_t
    hsa_soft_queue_create.argtypes = [hsa_region_t, uint32_t, hsa_queue_type32_t, uint32_t, hsa_signal_t, ctypes.POINTER(ctypes.POINTER(struct_hsa_queue_s))]
except AttributeError:
    pass
try:
    hsa_queue_destroy = _libraries['libhsa-runtime64.so'].hsa_queue_destroy
    hsa_queue_destroy.restype = hsa_status_t
    hsa_queue_destroy.argtypes = [ctypes.POINTER(struct_hsa_queue_s)]
except AttributeError:
    pass
try:
    hsa_queue_inactivate = _libraries['libhsa-runtime64.so'].hsa_queue_inactivate
    hsa_queue_inactivate.restype = hsa_status_t
    hsa_queue_inactivate.argtypes = [ctypes.POINTER(struct_hsa_queue_s)]
except AttributeError:
    pass
try:
    hsa_queue_load_read_index_acquire = _libraries['libhsa-runtime64.so'].hsa_queue_load_read_index_acquire
    hsa_queue_load_read_index_acquire.restype = uint64_t
    hsa_queue_load_read_index_acquire.argtypes = [ctypes.POINTER(struct_hsa_queue_s)]
except AttributeError:
    pass
try:
    hsa_queue_load_read_index_scacquire = _libraries['libhsa-runtime64.so'].hsa_queue_load_read_index_scacquire
    hsa_queue_load_read_index_scacquire.restype = uint64_t
    hsa_queue_load_read_index_scacquire.argtypes = [ctypes.POINTER(struct_hsa_queue_s)]
except AttributeError:
    pass
try:
    hsa_queue_load_read_index_relaxed = _libraries['libhsa-runtime64.so'].hsa_queue_load_read_index_relaxed
    hsa_queue_load_read_index_relaxed.restype = uint64_t
    hsa_queue_load_read_index_relaxed.argtypes = [ctypes.POINTER(struct_hsa_queue_s)]
except AttributeError:
    pass
try:
    hsa_queue_load_write_index_acquire = _libraries['libhsa-runtime64.so'].hsa_queue_load_write_index_acquire
    hsa_queue_load_write_index_acquire.restype = uint64_t
    hsa_queue_load_write_index_acquire.argtypes = [ctypes.POINTER(struct_hsa_queue_s)]
except AttributeError:
    pass
try:
    hsa_queue_load_write_index_scacquire = _libraries['libhsa-runtime64.so'].hsa_queue_load_write_index_scacquire
    hsa_queue_load_write_index_scacquire.restype = uint64_t
    hsa_queue_load_write_index_scacquire.argtypes = [ctypes.POINTER(struct_hsa_queue_s)]
except AttributeError:
    pass
try:
    hsa_queue_load_write_index_relaxed = _libraries['libhsa-runtime64.so'].hsa_queue_load_write_index_relaxed
    hsa_queue_load_write_index_relaxed.restype = uint64_t
    hsa_queue_load_write_index_relaxed.argtypes = [ctypes.POINTER(struct_hsa_queue_s)]
except AttributeError:
    pass
try:
    hsa_queue_store_write_index_relaxed = _libraries['libhsa-runtime64.so'].hsa_queue_store_write_index_relaxed
    hsa_queue_store_write_index_relaxed.restype = None
    hsa_queue_store_write_index_relaxed.argtypes = [ctypes.POINTER(struct_hsa_queue_s), uint64_t]
except AttributeError:
    pass
try:
    hsa_queue_store_write_index_release = _libraries['libhsa-runtime64.so'].hsa_queue_store_write_index_release
    hsa_queue_store_write_index_release.restype = None
    hsa_queue_store_write_index_release.argtypes = [ctypes.POINTER(struct_hsa_queue_s), uint64_t]
except AttributeError:
    pass
try:
    hsa_queue_store_write_index_screlease = _libraries['libhsa-runtime64.so'].hsa_queue_store_write_index_screlease
    hsa_queue_store_write_index_screlease.restype = None
    hsa_queue_store_write_index_screlease.argtypes = [ctypes.POINTER(struct_hsa_queue_s), uint64_t]
except AttributeError:
    pass
try:
    hsa_queue_cas_write_index_acq_rel = _libraries['libhsa-runtime64.so'].hsa_queue_cas_write_index_acq_rel
    hsa_queue_cas_write_index_acq_rel.restype = uint64_t
    hsa_queue_cas_write_index_acq_rel.argtypes = [ctypes.POINTER(struct_hsa_queue_s), uint64_t, uint64_t]
except AttributeError:
    pass
try:
    hsa_queue_cas_write_index_scacq_screl = _libraries['libhsa-runtime64.so'].hsa_queue_cas_write_index_scacq_screl
    hsa_queue_cas_write_index_scacq_screl.restype = uint64_t
    hsa_queue_cas_write_index_scacq_screl.argtypes = [ctypes.POINTER(struct_hsa_queue_s), uint64_t, uint64_t]
except AttributeError:
    pass
try:
    hsa_queue_cas_write_index_acquire = _libraries['libhsa-runtime64.so'].hsa_queue_cas_write_index_acquire
    hsa_queue_cas_write_index_acquire.restype = uint64_t
    hsa_queue_cas_write_index_acquire.argtypes = [ctypes.POINTER(struct_hsa_queue_s), uint64_t, uint64_t]
except AttributeError:
    pass
try:
    hsa_queue_cas_write_index_scacquire = _libraries['libhsa-runtime64.so'].hsa_queue_cas_write_index_scacquire
    hsa_queue_cas_write_index_scacquire.restype = uint64_t
    hsa_queue_cas_write_index_scacquire.argtypes = [ctypes.POINTER(struct_hsa_queue_s), uint64_t, uint64_t]
except AttributeError:
    pass
try:
    hsa_queue_cas_write_index_relaxed = _libraries['libhsa-runtime64.so'].hsa_queue_cas_write_index_relaxed
    hsa_queue_cas_write_index_relaxed.restype = uint64_t
    hsa_queue_cas_write_index_relaxed.argtypes = [ctypes.POINTER(struct_hsa_queue_s), uint64_t, uint64_t]
except AttributeError:
    pass
try:
    hsa_queue_cas_write_index_release = _libraries['libhsa-runtime64.so'].hsa_queue_cas_write_index_release
    hsa_queue_cas_write_index_release.restype = uint64_t
    hsa_queue_cas_write_index_release.argtypes = [ctypes.POINTER(struct_hsa_queue_s), uint64_t, uint64_t]
except AttributeError:
    pass
try:
    hsa_queue_cas_write_index_screlease = _libraries['libhsa-runtime64.so'].hsa_queue_cas_write_index_screlease
    hsa_queue_cas_write_index_screlease.restype = uint64_t
    hsa_queue_cas_write_index_screlease.argtypes = [ctypes.POINTER(struct_hsa_queue_s), uint64_t, uint64_t]
except AttributeError:
    pass
try:
    hsa_queue_add_write_index_acq_rel = _libraries['libhsa-runtime64.so'].hsa_queue_add_write_index_acq_rel
    hsa_queue_add_write_index_acq_rel.restype = uint64_t
    hsa_queue_add_write_index_acq_rel.argtypes = [ctypes.POINTER(struct_hsa_queue_s), uint64_t]
except AttributeError:
    pass
try:
    hsa_queue_add_write_index_scacq_screl = _libraries['libhsa-runtime64.so'].hsa_queue_add_write_index_scacq_screl
    hsa_queue_add_write_index_scacq_screl.restype = uint64_t
    hsa_queue_add_write_index_scacq_screl.argtypes = [ctypes.POINTER(struct_hsa_queue_s), uint64_t]
except AttributeError:
    pass
try:
    hsa_queue_add_write_index_acquire = _libraries['libhsa-runtime64.so'].hsa_queue_add_write_index_acquire
    hsa_queue_add_write_index_acquire.restype = uint64_t
    hsa_queue_add_write_index_acquire.argtypes = [ctypes.POINTER(struct_hsa_queue_s), uint64_t]
except AttributeError:
    pass
try:
    hsa_queue_add_write_index_scacquire = _libraries['libhsa-runtime64.so'].hsa_queue_add_write_index_scacquire
    hsa_queue_add_write_index_scacquire.restype = uint64_t
    hsa_queue_add_write_index_scacquire.argtypes = [ctypes.POINTER(struct_hsa_queue_s), uint64_t]
except AttributeError:
    pass
try:
    hsa_queue_add_write_index_relaxed = _libraries['libhsa-runtime64.so'].hsa_queue_add_write_index_relaxed
    hsa_queue_add_write_index_relaxed.restype = uint64_t
    hsa_queue_add_write_index_relaxed.argtypes = [ctypes.POINTER(struct_hsa_queue_s), uint64_t]
except AttributeError:
    pass
try:
    hsa_queue_add_write_index_release = _libraries['libhsa-runtime64.so'].hsa_queue_add_write_index_release
    hsa_queue_add_write_index_release.restype = uint64_t
    hsa_queue_add_write_index_release.argtypes = [ctypes.POINTER(struct_hsa_queue_s), uint64_t]
except AttributeError:
    pass
try:
    hsa_queue_add_write_index_screlease = _libraries['libhsa-runtime64.so'].hsa_queue_add_write_index_screlease
    hsa_queue_add_write_index_screlease.restype = uint64_t
    hsa_queue_add_write_index_screlease.argtypes = [ctypes.POINTER(struct_hsa_queue_s), uint64_t]
except AttributeError:
    pass
try:
    hsa_queue_store_read_index_relaxed = _libraries['libhsa-runtime64.so'].hsa_queue_store_read_index_relaxed
    hsa_queue_store_read_index_relaxed.restype = None
    hsa_queue_store_read_index_relaxed.argtypes = [ctypes.POINTER(struct_hsa_queue_s), uint64_t]
except AttributeError:
    pass
try:
    hsa_queue_store_read_index_release = _libraries['libhsa-runtime64.so'].hsa_queue_store_read_index_release
    hsa_queue_store_read_index_release.restype = None
    hsa_queue_store_read_index_release.argtypes = [ctypes.POINTER(struct_hsa_queue_s), uint64_t]
except AttributeError:
    pass
try:
    hsa_queue_store_read_index_screlease = _libraries['libhsa-runtime64.so'].hsa_queue_store_read_index_screlease
    hsa_queue_store_read_index_screlease.restype = None
    hsa_queue_store_read_index_screlease.argtypes = [ctypes.POINTER(struct_hsa_queue_s), uint64_t]
except AttributeError:
    pass

# values for enumeration 'hsa_packet_type_t'
hsa_packet_type_t__enumvalues = {
    0: 'HSA_PACKET_TYPE_VENDOR_SPECIFIC',
    1: 'HSA_PACKET_TYPE_INVALID',
    2: 'HSA_PACKET_TYPE_KERNEL_DISPATCH',
    3: 'HSA_PACKET_TYPE_BARRIER_AND',
    4: 'HSA_PACKET_TYPE_AGENT_DISPATCH',
    5: 'HSA_PACKET_TYPE_BARRIER_OR',
}
HSA_PACKET_TYPE_VENDOR_SPECIFIC = 0
HSA_PACKET_TYPE_INVALID = 1
HSA_PACKET_TYPE_KERNEL_DISPATCH = 2
HSA_PACKET_TYPE_BARRIER_AND = 3
HSA_PACKET_TYPE_AGENT_DISPATCH = 4
HSA_PACKET_TYPE_BARRIER_OR = 5
hsa_packet_type_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_fence_scope_t'
hsa_fence_scope_t__enumvalues = {
    0: 'HSA_FENCE_SCOPE_NONE',
    1: 'HSA_FENCE_SCOPE_AGENT',
    2: 'HSA_FENCE_SCOPE_SYSTEM',
}
HSA_FENCE_SCOPE_NONE = 0
HSA_FENCE_SCOPE_AGENT = 1
HSA_FENCE_SCOPE_SYSTEM = 2
hsa_fence_scope_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_packet_header_t'
hsa_packet_header_t__enumvalues = {
    0: 'HSA_PACKET_HEADER_TYPE',
    8: 'HSA_PACKET_HEADER_BARRIER',
    9: 'HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE',
    9: 'HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE',
    11: 'HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE',
    11: 'HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE',
}
HSA_PACKET_HEADER_TYPE = 0
HSA_PACKET_HEADER_BARRIER = 8
HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE = 9
HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE = 9
HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE = 11
HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE = 11
hsa_packet_header_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_packet_header_width_t'
hsa_packet_header_width_t__enumvalues = {
    8: 'HSA_PACKET_HEADER_WIDTH_TYPE',
    1: 'HSA_PACKET_HEADER_WIDTH_BARRIER',
    2: 'HSA_PACKET_HEADER_WIDTH_SCACQUIRE_FENCE_SCOPE',
    2: 'HSA_PACKET_HEADER_WIDTH_ACQUIRE_FENCE_SCOPE',
    2: 'HSA_PACKET_HEADER_WIDTH_SCRELEASE_FENCE_SCOPE',
    2: 'HSA_PACKET_HEADER_WIDTH_RELEASE_FENCE_SCOPE',
}
HSA_PACKET_HEADER_WIDTH_TYPE = 8
HSA_PACKET_HEADER_WIDTH_BARRIER = 1
HSA_PACKET_HEADER_WIDTH_SCACQUIRE_FENCE_SCOPE = 2
HSA_PACKET_HEADER_WIDTH_ACQUIRE_FENCE_SCOPE = 2
HSA_PACKET_HEADER_WIDTH_SCRELEASE_FENCE_SCOPE = 2
HSA_PACKET_HEADER_WIDTH_RELEASE_FENCE_SCOPE = 2
hsa_packet_header_width_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_kernel_dispatch_packet_setup_t'
hsa_kernel_dispatch_packet_setup_t__enumvalues = {
    0: 'HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS',
}
HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS = 0
hsa_kernel_dispatch_packet_setup_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_kernel_dispatch_packet_setup_width_t'
hsa_kernel_dispatch_packet_setup_width_t__enumvalues = {
    2: 'HSA_KERNEL_DISPATCH_PACKET_SETUP_WIDTH_DIMENSIONS',
}
HSA_KERNEL_DISPATCH_PACKET_SETUP_WIDTH_DIMENSIONS = 2
hsa_kernel_dispatch_packet_setup_width_t = ctypes.c_uint32 # enum
class struct_hsa_kernel_dispatch_packet_s(Structure):
    pass

struct_hsa_kernel_dispatch_packet_s._pack_ = 1 # source:False
struct_hsa_kernel_dispatch_packet_s._fields_ = [
    ('header', ctypes.c_uint16),
    ('setup', ctypes.c_uint16),
    ('workgroup_size_x', ctypes.c_uint16),
    ('workgroup_size_y', ctypes.c_uint16),
    ('workgroup_size_z', ctypes.c_uint16),
    ('reserved0', ctypes.c_uint16),
    ('grid_size_x', ctypes.c_uint32),
    ('grid_size_y', ctypes.c_uint32),
    ('grid_size_z', ctypes.c_uint32),
    ('private_segment_size', ctypes.c_uint32),
    ('group_segment_size', ctypes.c_uint32),
    ('kernel_object', ctypes.c_uint64),
    ('kernarg_address', ctypes.POINTER(None)),
    ('reserved2', ctypes.c_uint64),
    ('completion_signal', hsa_signal_t),
]

hsa_kernel_dispatch_packet_t = struct_hsa_kernel_dispatch_packet_s
class struct_hsa_agent_dispatch_packet_s(Structure):
    pass

struct_hsa_agent_dispatch_packet_s._pack_ = 1 # source:False
struct_hsa_agent_dispatch_packet_s._fields_ = [
    ('header', ctypes.c_uint16),
    ('type', ctypes.c_uint16),
    ('reserved0', ctypes.c_uint32),
    ('return_address', ctypes.POINTER(None)),
    ('arg', ctypes.c_uint64 * 4),
    ('reserved2', ctypes.c_uint64),
    ('completion_signal', hsa_signal_t),
]

hsa_agent_dispatch_packet_t = struct_hsa_agent_dispatch_packet_s
class struct_hsa_barrier_and_packet_s(Structure):
    pass

struct_hsa_barrier_and_packet_s._pack_ = 1 # source:False
struct_hsa_barrier_and_packet_s._fields_ = [
    ('header', ctypes.c_uint16),
    ('reserved0', ctypes.c_uint16),
    ('reserved1', ctypes.c_uint32),
    ('dep_signal', struct_hsa_signal_s * 5),
    ('reserved2', ctypes.c_uint64),
    ('completion_signal', hsa_signal_t),
]

hsa_barrier_and_packet_t = struct_hsa_barrier_and_packet_s
class struct_hsa_barrier_or_packet_s(Structure):
    pass

struct_hsa_barrier_or_packet_s._pack_ = 1 # source:False
struct_hsa_barrier_or_packet_s._fields_ = [
    ('header', ctypes.c_uint16),
    ('reserved0', ctypes.c_uint16),
    ('reserved1', ctypes.c_uint32),
    ('dep_signal', struct_hsa_signal_s * 5),
    ('reserved2', ctypes.c_uint64),
    ('completion_signal', hsa_signal_t),
]

hsa_barrier_or_packet_t = struct_hsa_barrier_or_packet_s

# values for enumeration 'hsa_region_segment_t'
hsa_region_segment_t__enumvalues = {
    0: 'HSA_REGION_SEGMENT_GLOBAL',
    1: 'HSA_REGION_SEGMENT_READONLY',
    2: 'HSA_REGION_SEGMENT_PRIVATE',
    3: 'HSA_REGION_SEGMENT_GROUP',
    4: 'HSA_REGION_SEGMENT_KERNARG',
}
HSA_REGION_SEGMENT_GLOBAL = 0
HSA_REGION_SEGMENT_READONLY = 1
HSA_REGION_SEGMENT_PRIVATE = 2
HSA_REGION_SEGMENT_GROUP = 3
HSA_REGION_SEGMENT_KERNARG = 4
hsa_region_segment_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_region_global_flag_t'
hsa_region_global_flag_t__enumvalues = {
    1: 'HSA_REGION_GLOBAL_FLAG_KERNARG',
    2: 'HSA_REGION_GLOBAL_FLAG_FINE_GRAINED',
    4: 'HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED',
    8: 'HSA_REGION_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED',
}
HSA_REGION_GLOBAL_FLAG_KERNARG = 1
HSA_REGION_GLOBAL_FLAG_FINE_GRAINED = 2
HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED = 4
HSA_REGION_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED = 8
hsa_region_global_flag_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_region_info_t'
hsa_region_info_t__enumvalues = {
    0: 'HSA_REGION_INFO_SEGMENT',
    1: 'HSA_REGION_INFO_GLOBAL_FLAGS',
    2: 'HSA_REGION_INFO_SIZE',
    4: 'HSA_REGION_INFO_ALLOC_MAX_SIZE',
    8: 'HSA_REGION_INFO_ALLOC_MAX_PRIVATE_WORKGROUP_SIZE',
    5: 'HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED',
    6: 'HSA_REGION_INFO_RUNTIME_ALLOC_GRANULE',
    7: 'HSA_REGION_INFO_RUNTIME_ALLOC_ALIGNMENT',
}
HSA_REGION_INFO_SEGMENT = 0
HSA_REGION_INFO_GLOBAL_FLAGS = 1
HSA_REGION_INFO_SIZE = 2
HSA_REGION_INFO_ALLOC_MAX_SIZE = 4
HSA_REGION_INFO_ALLOC_MAX_PRIVATE_WORKGROUP_SIZE = 8
HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED = 5
HSA_REGION_INFO_RUNTIME_ALLOC_GRANULE = 6
HSA_REGION_INFO_RUNTIME_ALLOC_ALIGNMENT = 7
hsa_region_info_t = ctypes.c_uint32 # enum
try:
    hsa_region_get_info = _libraries['libhsa-runtime64.so'].hsa_region_get_info
    hsa_region_get_info.restype = hsa_status_t
    hsa_region_get_info.argtypes = [hsa_region_t, hsa_region_info_t, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_agent_iterate_regions = _libraries['libhsa-runtime64.so'].hsa_agent_iterate_regions
    hsa_agent_iterate_regions.restype = hsa_status_t
    hsa_agent_iterate_regions.argtypes = [hsa_agent_t, ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_region_s, ctypes.POINTER(None)), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_memory_allocate = _libraries['libhsa-runtime64.so'].hsa_memory_allocate
    hsa_memory_allocate.restype = hsa_status_t
    hsa_memory_allocate.argtypes = [hsa_region_t, size_t, ctypes.POINTER(ctypes.POINTER(None))]
except AttributeError:
    pass
try:
    hsa_memory_free = _libraries['libhsa-runtime64.so'].hsa_memory_free
    hsa_memory_free.restype = hsa_status_t
    hsa_memory_free.argtypes = [ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_memory_copy = _libraries['libhsa-runtime64.so'].hsa_memory_copy
    hsa_memory_copy.restype = hsa_status_t
    hsa_memory_copy.argtypes = [ctypes.POINTER(None), ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    hsa_memory_assign_agent = _libraries['libhsa-runtime64.so'].hsa_memory_assign_agent
    hsa_memory_assign_agent.restype = hsa_status_t
    hsa_memory_assign_agent.argtypes = [ctypes.POINTER(None), hsa_agent_t, hsa_access_permission_t]
except AttributeError:
    pass
try:
    hsa_memory_register = _libraries['libhsa-runtime64.so'].hsa_memory_register
    hsa_memory_register.restype = hsa_status_t
    hsa_memory_register.argtypes = [ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    hsa_memory_deregister = _libraries['libhsa-runtime64.so'].hsa_memory_deregister
    hsa_memory_deregister.restype = hsa_status_t
    hsa_memory_deregister.argtypes = [ctypes.POINTER(None), size_t]
except AttributeError:
    pass
class struct_hsa_isa_s(Structure):
    pass

struct_hsa_isa_s._pack_ = 1 # source:False
struct_hsa_isa_s._fields_ = [
    ('handle', ctypes.c_uint64),
]

hsa_isa_t = struct_hsa_isa_s
try:
    hsa_isa_from_name = _libraries['libhsa-runtime64.so'].hsa_isa_from_name
    hsa_isa_from_name.restype = hsa_status_t
    hsa_isa_from_name.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct_hsa_isa_s)]
except AttributeError:
    pass
try:
    hsa_agent_iterate_isas = _libraries['libhsa-runtime64.so'].hsa_agent_iterate_isas
    hsa_agent_iterate_isas.restype = hsa_status_t
    hsa_agent_iterate_isas.argtypes = [hsa_agent_t, ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_isa_s, ctypes.POINTER(None)), ctypes.POINTER(None)]
except AttributeError:
    pass

# values for enumeration 'hsa_isa_info_t'
hsa_isa_info_t__enumvalues = {
    0: 'HSA_ISA_INFO_NAME_LENGTH',
    1: 'HSA_ISA_INFO_NAME',
    2: 'HSA_ISA_INFO_CALL_CONVENTION_COUNT',
    3: 'HSA_ISA_INFO_CALL_CONVENTION_INFO_WAVEFRONT_SIZE',
    4: 'HSA_ISA_INFO_CALL_CONVENTION_INFO_WAVEFRONTS_PER_COMPUTE_UNIT',
    5: 'HSA_ISA_INFO_MACHINE_MODELS',
    6: 'HSA_ISA_INFO_PROFILES',
    7: 'HSA_ISA_INFO_DEFAULT_FLOAT_ROUNDING_MODES',
    8: 'HSA_ISA_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES',
    9: 'HSA_ISA_INFO_FAST_F16_OPERATION',
    12: 'HSA_ISA_INFO_WORKGROUP_MAX_DIM',
    13: 'HSA_ISA_INFO_WORKGROUP_MAX_SIZE',
    14: 'HSA_ISA_INFO_GRID_MAX_DIM',
    16: 'HSA_ISA_INFO_GRID_MAX_SIZE',
    17: 'HSA_ISA_INFO_FBARRIER_MAX_SIZE',
}
HSA_ISA_INFO_NAME_LENGTH = 0
HSA_ISA_INFO_NAME = 1
HSA_ISA_INFO_CALL_CONVENTION_COUNT = 2
HSA_ISA_INFO_CALL_CONVENTION_INFO_WAVEFRONT_SIZE = 3
HSA_ISA_INFO_CALL_CONVENTION_INFO_WAVEFRONTS_PER_COMPUTE_UNIT = 4
HSA_ISA_INFO_MACHINE_MODELS = 5
HSA_ISA_INFO_PROFILES = 6
HSA_ISA_INFO_DEFAULT_FLOAT_ROUNDING_MODES = 7
HSA_ISA_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES = 8
HSA_ISA_INFO_FAST_F16_OPERATION = 9
HSA_ISA_INFO_WORKGROUP_MAX_DIM = 12
HSA_ISA_INFO_WORKGROUP_MAX_SIZE = 13
HSA_ISA_INFO_GRID_MAX_DIM = 14
HSA_ISA_INFO_GRID_MAX_SIZE = 16
HSA_ISA_INFO_FBARRIER_MAX_SIZE = 17
hsa_isa_info_t = ctypes.c_uint32 # enum
try:
    hsa_isa_get_info = _libraries['libhsa-runtime64.so'].hsa_isa_get_info
    hsa_isa_get_info.restype = hsa_status_t
    hsa_isa_get_info.argtypes = [hsa_isa_t, hsa_isa_info_t, uint32_t, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_isa_get_info_alt = _libraries['libhsa-runtime64.so'].hsa_isa_get_info_alt
    hsa_isa_get_info_alt.restype = hsa_status_t
    hsa_isa_get_info_alt.argtypes = [hsa_isa_t, hsa_isa_info_t, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_isa_get_exception_policies = _libraries['libhsa-runtime64.so'].hsa_isa_get_exception_policies
    hsa_isa_get_exception_policies.restype = hsa_status_t
    hsa_isa_get_exception_policies.argtypes = [hsa_isa_t, hsa_profile_t, ctypes.POINTER(ctypes.c_uint16)]
except AttributeError:
    pass

# values for enumeration 'hsa_fp_type_t'
hsa_fp_type_t__enumvalues = {
    1: 'HSA_FP_TYPE_16',
    2: 'HSA_FP_TYPE_32',
    4: 'HSA_FP_TYPE_64',
}
HSA_FP_TYPE_16 = 1
HSA_FP_TYPE_32 = 2
HSA_FP_TYPE_64 = 4
hsa_fp_type_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_flush_mode_t'
hsa_flush_mode_t__enumvalues = {
    1: 'HSA_FLUSH_MODE_FTZ',
    2: 'HSA_FLUSH_MODE_NON_FTZ',
}
HSA_FLUSH_MODE_FTZ = 1
HSA_FLUSH_MODE_NON_FTZ = 2
hsa_flush_mode_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_round_method_t'
hsa_round_method_t__enumvalues = {
    1: 'HSA_ROUND_METHOD_SINGLE',
    2: 'HSA_ROUND_METHOD_DOUBLE',
}
HSA_ROUND_METHOD_SINGLE = 1
HSA_ROUND_METHOD_DOUBLE = 2
hsa_round_method_t = ctypes.c_uint32 # enum
try:
    hsa_isa_get_round_method = _libraries['libhsa-runtime64.so'].hsa_isa_get_round_method
    hsa_isa_get_round_method.restype = hsa_status_t
    hsa_isa_get_round_method.argtypes = [hsa_isa_t, hsa_fp_type_t, hsa_flush_mode_t, ctypes.POINTER(hsa_round_method_t)]
except AttributeError:
    pass
class struct_hsa_wavefront_s(Structure):
    pass

struct_hsa_wavefront_s._pack_ = 1 # source:False
struct_hsa_wavefront_s._fields_ = [
    ('handle', ctypes.c_uint64),
]

hsa_wavefront_t = struct_hsa_wavefront_s

# values for enumeration 'hsa_wavefront_info_t'
hsa_wavefront_info_t__enumvalues = {
    0: 'HSA_WAVEFRONT_INFO_SIZE',
}
HSA_WAVEFRONT_INFO_SIZE = 0
hsa_wavefront_info_t = ctypes.c_uint32 # enum
try:
    hsa_wavefront_get_info = _libraries['libhsa-runtime64.so'].hsa_wavefront_get_info
    hsa_wavefront_get_info.restype = hsa_status_t
    hsa_wavefront_get_info.argtypes = [hsa_wavefront_t, hsa_wavefront_info_t, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_isa_iterate_wavefronts = _libraries['libhsa-runtime64.so'].hsa_isa_iterate_wavefronts
    hsa_isa_iterate_wavefronts.restype = hsa_status_t
    hsa_isa_iterate_wavefronts.argtypes = [hsa_isa_t, ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_wavefront_s, ctypes.POINTER(None)), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_isa_compatible = _libraries['libhsa-runtime64.so'].hsa_isa_compatible
    hsa_isa_compatible.restype = hsa_status_t
    hsa_isa_compatible.argtypes = [hsa_isa_t, hsa_isa_t, ctypes.POINTER(ctypes.c_bool)]
except AttributeError:
    pass
class struct_hsa_code_object_reader_s(Structure):
    pass

struct_hsa_code_object_reader_s._pack_ = 1 # source:False
struct_hsa_code_object_reader_s._fields_ = [
    ('handle', ctypes.c_uint64),
]

hsa_code_object_reader_t = struct_hsa_code_object_reader_s
try:
    hsa_code_object_reader_create_from_file = _libraries['libhsa-runtime64.so'].hsa_code_object_reader_create_from_file
    hsa_code_object_reader_create_from_file.restype = hsa_status_t
    hsa_code_object_reader_create_from_file.argtypes = [hsa_file_t, ctypes.POINTER(struct_hsa_code_object_reader_s)]
except AttributeError:
    pass
try:
    hsa_code_object_reader_create_from_memory = _libraries['libhsa-runtime64.so'].hsa_code_object_reader_create_from_memory
    hsa_code_object_reader_create_from_memory.restype = hsa_status_t
    hsa_code_object_reader_create_from_memory.argtypes = [ctypes.POINTER(None), size_t, ctypes.POINTER(struct_hsa_code_object_reader_s)]
except AttributeError:
    pass
try:
    hsa_code_object_reader_destroy = _libraries['libhsa-runtime64.so'].hsa_code_object_reader_destroy
    hsa_code_object_reader_destroy.restype = hsa_status_t
    hsa_code_object_reader_destroy.argtypes = [hsa_code_object_reader_t]
except AttributeError:
    pass
class struct_hsa_executable_s(Structure):
    pass

struct_hsa_executable_s._pack_ = 1 # source:False
struct_hsa_executable_s._fields_ = [
    ('handle', ctypes.c_uint64),
]

hsa_executable_t = struct_hsa_executable_s

# values for enumeration 'hsa_executable_state_t'
hsa_executable_state_t__enumvalues = {
    0: 'HSA_EXECUTABLE_STATE_UNFROZEN',
    1: 'HSA_EXECUTABLE_STATE_FROZEN',
}
HSA_EXECUTABLE_STATE_UNFROZEN = 0
HSA_EXECUTABLE_STATE_FROZEN = 1
hsa_executable_state_t = ctypes.c_uint32 # enum
try:
    hsa_executable_create = _libraries['libhsa-runtime64.so'].hsa_executable_create
    hsa_executable_create.restype = hsa_status_t
    hsa_executable_create.argtypes = [hsa_profile_t, hsa_executable_state_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct_hsa_executable_s)]
except AttributeError:
    pass
try:
    hsa_executable_create_alt = _libraries['libhsa-runtime64.so'].hsa_executable_create_alt
    hsa_executable_create_alt.restype = hsa_status_t
    hsa_executable_create_alt.argtypes = [hsa_profile_t, hsa_default_float_rounding_mode_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct_hsa_executable_s)]
except AttributeError:
    pass
try:
    hsa_executable_destroy = _libraries['libhsa-runtime64.so'].hsa_executable_destroy
    hsa_executable_destroy.restype = hsa_status_t
    hsa_executable_destroy.argtypes = [hsa_executable_t]
except AttributeError:
    pass
class struct_hsa_loaded_code_object_s(Structure):
    pass

struct_hsa_loaded_code_object_s._pack_ = 1 # source:False
struct_hsa_loaded_code_object_s._fields_ = [
    ('handle', ctypes.c_uint64),
]

hsa_loaded_code_object_t = struct_hsa_loaded_code_object_s
try:
    hsa_executable_load_program_code_object = _libraries['libhsa-runtime64.so'].hsa_executable_load_program_code_object
    hsa_executable_load_program_code_object.restype = hsa_status_t
    hsa_executable_load_program_code_object.argtypes = [hsa_executable_t, hsa_code_object_reader_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct_hsa_loaded_code_object_s)]
except AttributeError:
    pass
try:
    hsa_executable_load_agent_code_object = _libraries['libhsa-runtime64.so'].hsa_executable_load_agent_code_object
    hsa_executable_load_agent_code_object.restype = hsa_status_t
    hsa_executable_load_agent_code_object.argtypes = [hsa_executable_t, hsa_agent_t, hsa_code_object_reader_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct_hsa_loaded_code_object_s)]
except AttributeError:
    pass
try:
    hsa_executable_freeze = _libraries['libhsa-runtime64.so'].hsa_executable_freeze
    hsa_executable_freeze.restype = hsa_status_t
    hsa_executable_freeze.argtypes = [hsa_executable_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass

# values for enumeration 'hsa_executable_info_t'
hsa_executable_info_t__enumvalues = {
    1: 'HSA_EXECUTABLE_INFO_PROFILE',
    2: 'HSA_EXECUTABLE_INFO_STATE',
    3: 'HSA_EXECUTABLE_INFO_DEFAULT_FLOAT_ROUNDING_MODE',
}
HSA_EXECUTABLE_INFO_PROFILE = 1
HSA_EXECUTABLE_INFO_STATE = 2
HSA_EXECUTABLE_INFO_DEFAULT_FLOAT_ROUNDING_MODE = 3
hsa_executable_info_t = ctypes.c_uint32 # enum
try:
    hsa_executable_get_info = _libraries['libhsa-runtime64.so'].hsa_executable_get_info
    hsa_executable_get_info.restype = hsa_status_t
    hsa_executable_get_info.argtypes = [hsa_executable_t, hsa_executable_info_t, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_executable_global_variable_define = _libraries['libhsa-runtime64.so'].hsa_executable_global_variable_define
    hsa_executable_global_variable_define.restype = hsa_status_t
    hsa_executable_global_variable_define.argtypes = [hsa_executable_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_executable_agent_global_variable_define = _libraries['libhsa-runtime64.so'].hsa_executable_agent_global_variable_define
    hsa_executable_agent_global_variable_define.restype = hsa_status_t
    hsa_executable_agent_global_variable_define.argtypes = [hsa_executable_t, hsa_agent_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_executable_readonly_variable_define = _libraries['libhsa-runtime64.so'].hsa_executable_readonly_variable_define
    hsa_executable_readonly_variable_define.restype = hsa_status_t
    hsa_executable_readonly_variable_define.argtypes = [hsa_executable_t, hsa_agent_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_executable_validate = _libraries['libhsa-runtime64.so'].hsa_executable_validate
    hsa_executable_validate.restype = hsa_status_t
    hsa_executable_validate.argtypes = [hsa_executable_t, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    hsa_executable_validate_alt = _libraries['libhsa-runtime64.so'].hsa_executable_validate_alt
    hsa_executable_validate_alt.restype = hsa_status_t
    hsa_executable_validate_alt.argtypes = [hsa_executable_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
class struct_hsa_executable_symbol_s(Structure):
    pass

struct_hsa_executable_symbol_s._pack_ = 1 # source:False
struct_hsa_executable_symbol_s._fields_ = [
    ('handle', ctypes.c_uint64),
]

hsa_executable_symbol_t = struct_hsa_executable_symbol_s
int32_t = ctypes.c_int32
try:
    hsa_executable_get_symbol = _libraries['libhsa-runtime64.so'].hsa_executable_get_symbol
    hsa_executable_get_symbol.restype = hsa_status_t
    hsa_executable_get_symbol.argtypes = [hsa_executable_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), hsa_agent_t, int32_t, ctypes.POINTER(struct_hsa_executable_symbol_s)]
except AttributeError:
    pass
try:
    hsa_executable_get_symbol_by_name = _libraries['libhsa-runtime64.so'].hsa_executable_get_symbol_by_name
    hsa_executable_get_symbol_by_name.restype = hsa_status_t
    hsa_executable_get_symbol_by_name.argtypes = [hsa_executable_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct_hsa_agent_s), ctypes.POINTER(struct_hsa_executable_symbol_s)]
except AttributeError:
    pass

# values for enumeration 'hsa_symbol_kind_t'
hsa_symbol_kind_t__enumvalues = {
    0: 'HSA_SYMBOL_KIND_VARIABLE',
    1: 'HSA_SYMBOL_KIND_KERNEL',
    2: 'HSA_SYMBOL_KIND_INDIRECT_FUNCTION',
}
HSA_SYMBOL_KIND_VARIABLE = 0
HSA_SYMBOL_KIND_KERNEL = 1
HSA_SYMBOL_KIND_INDIRECT_FUNCTION = 2
hsa_symbol_kind_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_symbol_linkage_t'
hsa_symbol_linkage_t__enumvalues = {
    0: 'HSA_SYMBOL_LINKAGE_MODULE',
    1: 'HSA_SYMBOL_LINKAGE_PROGRAM',
}
HSA_SYMBOL_LINKAGE_MODULE = 0
HSA_SYMBOL_LINKAGE_PROGRAM = 1
hsa_symbol_linkage_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_variable_allocation_t'
hsa_variable_allocation_t__enumvalues = {
    0: 'HSA_VARIABLE_ALLOCATION_AGENT',
    1: 'HSA_VARIABLE_ALLOCATION_PROGRAM',
}
HSA_VARIABLE_ALLOCATION_AGENT = 0
HSA_VARIABLE_ALLOCATION_PROGRAM = 1
hsa_variable_allocation_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_variable_segment_t'
hsa_variable_segment_t__enumvalues = {
    0: 'HSA_VARIABLE_SEGMENT_GLOBAL',
    1: 'HSA_VARIABLE_SEGMENT_READONLY',
}
HSA_VARIABLE_SEGMENT_GLOBAL = 0
HSA_VARIABLE_SEGMENT_READONLY = 1
hsa_variable_segment_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_executable_symbol_info_t'
hsa_executable_symbol_info_t__enumvalues = {
    0: 'HSA_EXECUTABLE_SYMBOL_INFO_TYPE',
    1: 'HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH',
    2: 'HSA_EXECUTABLE_SYMBOL_INFO_NAME',
    3: 'HSA_EXECUTABLE_SYMBOL_INFO_MODULE_NAME_LENGTH',
    4: 'HSA_EXECUTABLE_SYMBOL_INFO_MODULE_NAME',
    20: 'HSA_EXECUTABLE_SYMBOL_INFO_AGENT',
    21: 'HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS',
    5: 'HSA_EXECUTABLE_SYMBOL_INFO_LINKAGE',
    17: 'HSA_EXECUTABLE_SYMBOL_INFO_IS_DEFINITION',
    6: 'HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ALLOCATION',
    7: 'HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SEGMENT',
    8: 'HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ALIGNMENT',
    9: 'HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SIZE',
    10: 'HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_IS_CONST',
    22: 'HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT',
    11: 'HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE',
    12: 'HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT',
    13: 'HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE',
    14: 'HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE',
    15: 'HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK',
    18: 'HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_CALL_CONVENTION',
    23: 'HSA_EXECUTABLE_SYMBOL_INFO_INDIRECT_FUNCTION_OBJECT',
    16: 'HSA_EXECUTABLE_SYMBOL_INFO_INDIRECT_FUNCTION_CALL_CONVENTION',
}
HSA_EXECUTABLE_SYMBOL_INFO_TYPE = 0
HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH = 1
HSA_EXECUTABLE_SYMBOL_INFO_NAME = 2
HSA_EXECUTABLE_SYMBOL_INFO_MODULE_NAME_LENGTH = 3
HSA_EXECUTABLE_SYMBOL_INFO_MODULE_NAME = 4
HSA_EXECUTABLE_SYMBOL_INFO_AGENT = 20
HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS = 21
HSA_EXECUTABLE_SYMBOL_INFO_LINKAGE = 5
HSA_EXECUTABLE_SYMBOL_INFO_IS_DEFINITION = 17
HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ALLOCATION = 6
HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SEGMENT = 7
HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ALIGNMENT = 8
HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SIZE = 9
HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_IS_CONST = 10
HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT = 22
HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE = 11
HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT = 12
HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE = 13
HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE = 14
HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK = 15
HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_CALL_CONVENTION = 18
HSA_EXECUTABLE_SYMBOL_INFO_INDIRECT_FUNCTION_OBJECT = 23
HSA_EXECUTABLE_SYMBOL_INFO_INDIRECT_FUNCTION_CALL_CONVENTION = 16
hsa_executable_symbol_info_t = ctypes.c_uint32 # enum
try:
    hsa_executable_symbol_get_info = _libraries['libhsa-runtime64.so'].hsa_executable_symbol_get_info
    hsa_executable_symbol_get_info.restype = hsa_status_t
    hsa_executable_symbol_get_info.argtypes = [hsa_executable_symbol_t, hsa_executable_symbol_info_t, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_executable_iterate_symbols = _libraries['libhsa-runtime64.so'].hsa_executable_iterate_symbols
    hsa_executable_iterate_symbols.restype = hsa_status_t
    hsa_executable_iterate_symbols.argtypes = [hsa_executable_t, ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_executable_s, struct_hsa_executable_symbol_s, ctypes.POINTER(None)), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_executable_iterate_agent_symbols = _libraries['libhsa-runtime64.so'].hsa_executable_iterate_agent_symbols
    hsa_executable_iterate_agent_symbols.restype = hsa_status_t
    hsa_executable_iterate_agent_symbols.argtypes = [hsa_executable_t, hsa_agent_t, ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_executable_s, struct_hsa_agent_s, struct_hsa_executable_symbol_s, ctypes.POINTER(None)), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_executable_iterate_program_symbols = _libraries['libhsa-runtime64.so'].hsa_executable_iterate_program_symbols
    hsa_executable_iterate_program_symbols.restype = hsa_status_t
    hsa_executable_iterate_program_symbols.argtypes = [hsa_executable_t, ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_executable_s, struct_hsa_executable_symbol_s, ctypes.POINTER(None)), ctypes.POINTER(None)]
except AttributeError:
    pass
class struct_hsa_code_object_s(Structure):
    pass

struct_hsa_code_object_s._pack_ = 1 # source:False
struct_hsa_code_object_s._fields_ = [
    ('handle', ctypes.c_uint64),
]

hsa_code_object_t = struct_hsa_code_object_s
class struct_hsa_callback_data_s(Structure):
    pass

struct_hsa_callback_data_s._pack_ = 1 # source:False
struct_hsa_callback_data_s._fields_ = [
    ('handle', ctypes.c_uint64),
]

hsa_callback_data_t = struct_hsa_callback_data_s
try:
    hsa_code_object_serialize = _libraries['libhsa-runtime64.so'].hsa_code_object_serialize
    hsa_code_object_serialize.restype = hsa_status_t
    hsa_code_object_serialize.argtypes = [hsa_code_object_t, ctypes.CFUNCTYPE(hsa_status_t, ctypes.c_uint64, struct_hsa_callback_data_s, ctypes.POINTER(ctypes.POINTER(None))), hsa_callback_data_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    hsa_code_object_deserialize = _libraries['libhsa-runtime64.so'].hsa_code_object_deserialize
    hsa_code_object_deserialize.restype = hsa_status_t
    hsa_code_object_deserialize.argtypes = [ctypes.POINTER(None), size_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct_hsa_code_object_s)]
except AttributeError:
    pass
try:
    hsa_code_object_destroy = _libraries['libhsa-runtime64.so'].hsa_code_object_destroy
    hsa_code_object_destroy.restype = hsa_status_t
    hsa_code_object_destroy.argtypes = [hsa_code_object_t]
except AttributeError:
    pass

# values for enumeration 'hsa_code_object_type_t'
hsa_code_object_type_t__enumvalues = {
    0: 'HSA_CODE_OBJECT_TYPE_PROGRAM',
}
HSA_CODE_OBJECT_TYPE_PROGRAM = 0
hsa_code_object_type_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_code_object_info_t'
hsa_code_object_info_t__enumvalues = {
    0: 'HSA_CODE_OBJECT_INFO_VERSION',
    1: 'HSA_CODE_OBJECT_INFO_TYPE',
    2: 'HSA_CODE_OBJECT_INFO_ISA',
    3: 'HSA_CODE_OBJECT_INFO_MACHINE_MODEL',
    4: 'HSA_CODE_OBJECT_INFO_PROFILE',
    5: 'HSA_CODE_OBJECT_INFO_DEFAULT_FLOAT_ROUNDING_MODE',
}
HSA_CODE_OBJECT_INFO_VERSION = 0
HSA_CODE_OBJECT_INFO_TYPE = 1
HSA_CODE_OBJECT_INFO_ISA = 2
HSA_CODE_OBJECT_INFO_MACHINE_MODEL = 3
HSA_CODE_OBJECT_INFO_PROFILE = 4
HSA_CODE_OBJECT_INFO_DEFAULT_FLOAT_ROUNDING_MODE = 5
hsa_code_object_info_t = ctypes.c_uint32 # enum
try:
    hsa_code_object_get_info = _libraries['libhsa-runtime64.so'].hsa_code_object_get_info
    hsa_code_object_get_info.restype = hsa_status_t
    hsa_code_object_get_info.argtypes = [hsa_code_object_t, hsa_code_object_info_t, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_executable_load_code_object = _libraries['libhsa-runtime64.so'].hsa_executable_load_code_object
    hsa_executable_load_code_object.restype = hsa_status_t
    hsa_executable_load_code_object.argtypes = [hsa_executable_t, hsa_agent_t, hsa_code_object_t, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
class struct_hsa_code_symbol_s(Structure):
    pass

struct_hsa_code_symbol_s._pack_ = 1 # source:False
struct_hsa_code_symbol_s._fields_ = [
    ('handle', ctypes.c_uint64),
]

hsa_code_symbol_t = struct_hsa_code_symbol_s
try:
    hsa_code_object_get_symbol = _libraries['libhsa-runtime64.so'].hsa_code_object_get_symbol
    hsa_code_object_get_symbol.restype = hsa_status_t
    hsa_code_object_get_symbol.argtypes = [hsa_code_object_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct_hsa_code_symbol_s)]
except AttributeError:
    pass
try:
    hsa_code_object_get_symbol_from_name = _libraries['libhsa-runtime64.so'].hsa_code_object_get_symbol_from_name
    hsa_code_object_get_symbol_from_name.restype = hsa_status_t
    hsa_code_object_get_symbol_from_name.argtypes = [hsa_code_object_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct_hsa_code_symbol_s)]
except AttributeError:
    pass

# values for enumeration 'hsa_code_symbol_info_t'
hsa_code_symbol_info_t__enumvalues = {
    0: 'HSA_CODE_SYMBOL_INFO_TYPE',
    1: 'HSA_CODE_SYMBOL_INFO_NAME_LENGTH',
    2: 'HSA_CODE_SYMBOL_INFO_NAME',
    3: 'HSA_CODE_SYMBOL_INFO_MODULE_NAME_LENGTH',
    4: 'HSA_CODE_SYMBOL_INFO_MODULE_NAME',
    5: 'HSA_CODE_SYMBOL_INFO_LINKAGE',
    17: 'HSA_CODE_SYMBOL_INFO_IS_DEFINITION',
    6: 'HSA_CODE_SYMBOL_INFO_VARIABLE_ALLOCATION',
    7: 'HSA_CODE_SYMBOL_INFO_VARIABLE_SEGMENT',
    8: 'HSA_CODE_SYMBOL_INFO_VARIABLE_ALIGNMENT',
    9: 'HSA_CODE_SYMBOL_INFO_VARIABLE_SIZE',
    10: 'HSA_CODE_SYMBOL_INFO_VARIABLE_IS_CONST',
    11: 'HSA_CODE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE',
    12: 'HSA_CODE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT',
    13: 'HSA_CODE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE',
    14: 'HSA_CODE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE',
    15: 'HSA_CODE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK',
    18: 'HSA_CODE_SYMBOL_INFO_KERNEL_CALL_CONVENTION',
    16: 'HSA_CODE_SYMBOL_INFO_INDIRECT_FUNCTION_CALL_CONVENTION',
}
HSA_CODE_SYMBOL_INFO_TYPE = 0
HSA_CODE_SYMBOL_INFO_NAME_LENGTH = 1
HSA_CODE_SYMBOL_INFO_NAME = 2
HSA_CODE_SYMBOL_INFO_MODULE_NAME_LENGTH = 3
HSA_CODE_SYMBOL_INFO_MODULE_NAME = 4
HSA_CODE_SYMBOL_INFO_LINKAGE = 5
HSA_CODE_SYMBOL_INFO_IS_DEFINITION = 17
HSA_CODE_SYMBOL_INFO_VARIABLE_ALLOCATION = 6
HSA_CODE_SYMBOL_INFO_VARIABLE_SEGMENT = 7
HSA_CODE_SYMBOL_INFO_VARIABLE_ALIGNMENT = 8
HSA_CODE_SYMBOL_INFO_VARIABLE_SIZE = 9
HSA_CODE_SYMBOL_INFO_VARIABLE_IS_CONST = 10
HSA_CODE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE = 11
HSA_CODE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT = 12
HSA_CODE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE = 13
HSA_CODE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE = 14
HSA_CODE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK = 15
HSA_CODE_SYMBOL_INFO_KERNEL_CALL_CONVENTION = 18
HSA_CODE_SYMBOL_INFO_INDIRECT_FUNCTION_CALL_CONVENTION = 16
hsa_code_symbol_info_t = ctypes.c_uint32 # enum
try:
    hsa_code_symbol_get_info = _libraries['libhsa-runtime64.so'].hsa_code_symbol_get_info
    hsa_code_symbol_get_info.restype = hsa_status_t
    hsa_code_symbol_get_info.argtypes = [hsa_code_symbol_t, hsa_code_symbol_info_t, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_code_object_iterate_symbols = _libraries['libhsa-runtime64.so'].hsa_code_object_iterate_symbols
    hsa_code_object_iterate_symbols.restype = hsa_status_t
    hsa_code_object_iterate_symbols.argtypes = [hsa_code_object_t, ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_code_object_s, struct_hsa_code_symbol_s, ctypes.POINTER(None)), ctypes.POINTER(None)]
except AttributeError:
    pass

# values for enumeration 'enum_hsa_ext_image_h_68'
enum_hsa_ext_image_h_68__enumvalues = {
    12288: 'HSA_EXT_STATUS_ERROR_IMAGE_FORMAT_UNSUPPORTED',
    12289: 'HSA_EXT_STATUS_ERROR_IMAGE_SIZE_UNSUPPORTED',
    12290: 'HSA_EXT_STATUS_ERROR_IMAGE_PITCH_UNSUPPORTED',
    12291: 'HSA_EXT_STATUS_ERROR_SAMPLER_DESCRIPTOR_UNSUPPORTED',
}
HSA_EXT_STATUS_ERROR_IMAGE_FORMAT_UNSUPPORTED = 12288
HSA_EXT_STATUS_ERROR_IMAGE_SIZE_UNSUPPORTED = 12289
HSA_EXT_STATUS_ERROR_IMAGE_PITCH_UNSUPPORTED = 12290
HSA_EXT_STATUS_ERROR_SAMPLER_DESCRIPTOR_UNSUPPORTED = 12291
enum_hsa_ext_image_h_68 = ctypes.c_uint32 # enum

# values for enumeration 'enum_hsa_ext_image_h_93'
enum_hsa_ext_image_h_93__enumvalues = {
    12288: 'HSA_EXT_AGENT_INFO_IMAGE_1D_MAX_ELEMENTS',
    12289: 'HSA_EXT_AGENT_INFO_IMAGE_1DA_MAX_ELEMENTS',
    12290: 'HSA_EXT_AGENT_INFO_IMAGE_1DB_MAX_ELEMENTS',
    12291: 'HSA_EXT_AGENT_INFO_IMAGE_2D_MAX_ELEMENTS',
    12292: 'HSA_EXT_AGENT_INFO_IMAGE_2DA_MAX_ELEMENTS',
    12293: 'HSA_EXT_AGENT_INFO_IMAGE_2DDEPTH_MAX_ELEMENTS',
    12294: 'HSA_EXT_AGENT_INFO_IMAGE_2DADEPTH_MAX_ELEMENTS',
    12295: 'HSA_EXT_AGENT_INFO_IMAGE_3D_MAX_ELEMENTS',
    12296: 'HSA_EXT_AGENT_INFO_IMAGE_ARRAY_MAX_LAYERS',
    12297: 'HSA_EXT_AGENT_INFO_MAX_IMAGE_RD_HANDLES',
    12298: 'HSA_EXT_AGENT_INFO_MAX_IMAGE_RORW_HANDLES',
    12299: 'HSA_EXT_AGENT_INFO_MAX_SAMPLER_HANDLERS',
    12300: 'HSA_EXT_AGENT_INFO_IMAGE_LINEAR_ROW_PITCH_ALIGNMENT',
}
HSA_EXT_AGENT_INFO_IMAGE_1D_MAX_ELEMENTS = 12288
HSA_EXT_AGENT_INFO_IMAGE_1DA_MAX_ELEMENTS = 12289
HSA_EXT_AGENT_INFO_IMAGE_1DB_MAX_ELEMENTS = 12290
HSA_EXT_AGENT_INFO_IMAGE_2D_MAX_ELEMENTS = 12291
HSA_EXT_AGENT_INFO_IMAGE_2DA_MAX_ELEMENTS = 12292
HSA_EXT_AGENT_INFO_IMAGE_2DDEPTH_MAX_ELEMENTS = 12293
HSA_EXT_AGENT_INFO_IMAGE_2DADEPTH_MAX_ELEMENTS = 12294
HSA_EXT_AGENT_INFO_IMAGE_3D_MAX_ELEMENTS = 12295
HSA_EXT_AGENT_INFO_IMAGE_ARRAY_MAX_LAYERS = 12296
HSA_EXT_AGENT_INFO_MAX_IMAGE_RD_HANDLES = 12297
HSA_EXT_AGENT_INFO_MAX_IMAGE_RORW_HANDLES = 12298
HSA_EXT_AGENT_INFO_MAX_SAMPLER_HANDLERS = 12299
HSA_EXT_AGENT_INFO_IMAGE_LINEAR_ROW_PITCH_ALIGNMENT = 12300
enum_hsa_ext_image_h_93 = ctypes.c_uint32 # enum
class struct_hsa_ext_image_s(Structure):
    pass

struct_hsa_ext_image_s._pack_ = 1 # source:False
struct_hsa_ext_image_s._fields_ = [
    ('handle', ctypes.c_uint64),
]

hsa_ext_image_t = struct_hsa_ext_image_s

# values for enumeration 'hsa_ext_image_geometry_t'
hsa_ext_image_geometry_t__enumvalues = {
    0: 'HSA_EXT_IMAGE_GEOMETRY_1D',
    1: 'HSA_EXT_IMAGE_GEOMETRY_2D',
    2: 'HSA_EXT_IMAGE_GEOMETRY_3D',
    3: 'HSA_EXT_IMAGE_GEOMETRY_1DA',
    4: 'HSA_EXT_IMAGE_GEOMETRY_2DA',
    5: 'HSA_EXT_IMAGE_GEOMETRY_1DB',
    6: 'HSA_EXT_IMAGE_GEOMETRY_2DDEPTH',
    7: 'HSA_EXT_IMAGE_GEOMETRY_2DADEPTH',
}
HSA_EXT_IMAGE_GEOMETRY_1D = 0
HSA_EXT_IMAGE_GEOMETRY_2D = 1
HSA_EXT_IMAGE_GEOMETRY_3D = 2
HSA_EXT_IMAGE_GEOMETRY_1DA = 3
HSA_EXT_IMAGE_GEOMETRY_2DA = 4
HSA_EXT_IMAGE_GEOMETRY_1DB = 5
HSA_EXT_IMAGE_GEOMETRY_2DDEPTH = 6
HSA_EXT_IMAGE_GEOMETRY_2DADEPTH = 7
hsa_ext_image_geometry_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_ext_image_channel_type_t'
hsa_ext_image_channel_type_t__enumvalues = {
    0: 'HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT8',
    1: 'HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT16',
    2: 'HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT8',
    3: 'HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT16',
    4: 'HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT24',
    5: 'HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_555',
    6: 'HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_565',
    7: 'HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_101010',
    8: 'HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT8',
    9: 'HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT16',
    10: 'HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT32',
    11: 'HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8',
    12: 'HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16',
    13: 'HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32',
    14: 'HSA_EXT_IMAGE_CHANNEL_TYPE_HALF_FLOAT',
    15: 'HSA_EXT_IMAGE_CHANNEL_TYPE_FLOAT',
}
HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT8 = 0
HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT16 = 1
HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT8 = 2
HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT16 = 3
HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT24 = 4
HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_555 = 5
HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_565 = 6
HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_101010 = 7
HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT8 = 8
HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT16 = 9
HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT32 = 10
HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8 = 11
HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16 = 12
HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32 = 13
HSA_EXT_IMAGE_CHANNEL_TYPE_HALF_FLOAT = 14
HSA_EXT_IMAGE_CHANNEL_TYPE_FLOAT = 15
hsa_ext_image_channel_type_t = ctypes.c_uint32 # enum
hsa_ext_image_channel_type32_t = ctypes.c_uint32

# values for enumeration 'hsa_ext_image_channel_order_t'
hsa_ext_image_channel_order_t__enumvalues = {
    0: 'HSA_EXT_IMAGE_CHANNEL_ORDER_A',
    1: 'HSA_EXT_IMAGE_CHANNEL_ORDER_R',
    2: 'HSA_EXT_IMAGE_CHANNEL_ORDER_RX',
    3: 'HSA_EXT_IMAGE_CHANNEL_ORDER_RG',
    4: 'HSA_EXT_IMAGE_CHANNEL_ORDER_RGX',
    5: 'HSA_EXT_IMAGE_CHANNEL_ORDER_RA',
    6: 'HSA_EXT_IMAGE_CHANNEL_ORDER_RGB',
    7: 'HSA_EXT_IMAGE_CHANNEL_ORDER_RGBX',
    8: 'HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA',
    9: 'HSA_EXT_IMAGE_CHANNEL_ORDER_BGRA',
    10: 'HSA_EXT_IMAGE_CHANNEL_ORDER_ARGB',
    11: 'HSA_EXT_IMAGE_CHANNEL_ORDER_ABGR',
    12: 'HSA_EXT_IMAGE_CHANNEL_ORDER_SRGB',
    13: 'HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBX',
    14: 'HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBA',
    15: 'HSA_EXT_IMAGE_CHANNEL_ORDER_SBGRA',
    16: 'HSA_EXT_IMAGE_CHANNEL_ORDER_INTENSITY',
    17: 'HSA_EXT_IMAGE_CHANNEL_ORDER_LUMINANCE',
    18: 'HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH',
    19: 'HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH_STENCIL',
}
HSA_EXT_IMAGE_CHANNEL_ORDER_A = 0
HSA_EXT_IMAGE_CHANNEL_ORDER_R = 1
HSA_EXT_IMAGE_CHANNEL_ORDER_RX = 2
HSA_EXT_IMAGE_CHANNEL_ORDER_RG = 3
HSA_EXT_IMAGE_CHANNEL_ORDER_RGX = 4
HSA_EXT_IMAGE_CHANNEL_ORDER_RA = 5
HSA_EXT_IMAGE_CHANNEL_ORDER_RGB = 6
HSA_EXT_IMAGE_CHANNEL_ORDER_RGBX = 7
HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA = 8
HSA_EXT_IMAGE_CHANNEL_ORDER_BGRA = 9
HSA_EXT_IMAGE_CHANNEL_ORDER_ARGB = 10
HSA_EXT_IMAGE_CHANNEL_ORDER_ABGR = 11
HSA_EXT_IMAGE_CHANNEL_ORDER_SRGB = 12
HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBX = 13
HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBA = 14
HSA_EXT_IMAGE_CHANNEL_ORDER_SBGRA = 15
HSA_EXT_IMAGE_CHANNEL_ORDER_INTENSITY = 16
HSA_EXT_IMAGE_CHANNEL_ORDER_LUMINANCE = 17
HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH = 18
HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH_STENCIL = 19
hsa_ext_image_channel_order_t = ctypes.c_uint32 # enum
hsa_ext_image_channel_order32_t = ctypes.c_uint32
class struct_hsa_ext_image_format_s(Structure):
    pass

struct_hsa_ext_image_format_s._pack_ = 1 # source:False
struct_hsa_ext_image_format_s._fields_ = [
    ('channel_type', ctypes.c_uint32),
    ('channel_order', ctypes.c_uint32),
]

hsa_ext_image_format_t = struct_hsa_ext_image_format_s
class struct_hsa_ext_image_descriptor_s(Structure):
    pass

struct_hsa_ext_image_descriptor_s._pack_ = 1 # source:False
struct_hsa_ext_image_descriptor_s._fields_ = [
    ('geometry', hsa_ext_image_geometry_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('width', ctypes.c_uint64),
    ('height', ctypes.c_uint64),
    ('depth', ctypes.c_uint64),
    ('array_size', ctypes.c_uint64),
    ('format', hsa_ext_image_format_t),
]

hsa_ext_image_descriptor_t = struct_hsa_ext_image_descriptor_s

# values for enumeration 'hsa_ext_image_capability_t'
hsa_ext_image_capability_t__enumvalues = {
    0: 'HSA_EXT_IMAGE_CAPABILITY_NOT_SUPPORTED',
    1: 'HSA_EXT_IMAGE_CAPABILITY_READ_ONLY',
    2: 'HSA_EXT_IMAGE_CAPABILITY_WRITE_ONLY',
    4: 'HSA_EXT_IMAGE_CAPABILITY_READ_WRITE',
    8: 'HSA_EXT_IMAGE_CAPABILITY_READ_MODIFY_WRITE',
    16: 'HSA_EXT_IMAGE_CAPABILITY_ACCESS_INVARIANT_DATA_LAYOUT',
}
HSA_EXT_IMAGE_CAPABILITY_NOT_SUPPORTED = 0
HSA_EXT_IMAGE_CAPABILITY_READ_ONLY = 1
HSA_EXT_IMAGE_CAPABILITY_WRITE_ONLY = 2
HSA_EXT_IMAGE_CAPABILITY_READ_WRITE = 4
HSA_EXT_IMAGE_CAPABILITY_READ_MODIFY_WRITE = 8
HSA_EXT_IMAGE_CAPABILITY_ACCESS_INVARIANT_DATA_LAYOUT = 16
hsa_ext_image_capability_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_ext_image_data_layout_t'
hsa_ext_image_data_layout_t__enumvalues = {
    0: 'HSA_EXT_IMAGE_DATA_LAYOUT_OPAQUE',
    1: 'HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR',
}
HSA_EXT_IMAGE_DATA_LAYOUT_OPAQUE = 0
HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR = 1
hsa_ext_image_data_layout_t = ctypes.c_uint32 # enum
try:
    hsa_ext_image_get_capability = _libraries['libhsa-runtime64.so'].hsa_ext_image_get_capability
    hsa_ext_image_get_capability.restype = hsa_status_t
    hsa_ext_image_get_capability.argtypes = [hsa_agent_t, hsa_ext_image_geometry_t, ctypes.POINTER(struct_hsa_ext_image_format_s), ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    hsa_ext_image_get_capability_with_layout = _libraries['libhsa-runtime64.so'].hsa_ext_image_get_capability_with_layout
    hsa_ext_image_get_capability_with_layout.restype = hsa_status_t
    hsa_ext_image_get_capability_with_layout.argtypes = [hsa_agent_t, hsa_ext_image_geometry_t, ctypes.POINTER(struct_hsa_ext_image_format_s), hsa_ext_image_data_layout_t, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
class struct_hsa_ext_image_data_info_s(Structure):
    pass

struct_hsa_ext_image_data_info_s._pack_ = 1 # source:False
struct_hsa_ext_image_data_info_s._fields_ = [
    ('size', ctypes.c_uint64),
    ('alignment', ctypes.c_uint64),
]

hsa_ext_image_data_info_t = struct_hsa_ext_image_data_info_s
try:
    hsa_ext_image_data_get_info = _libraries['libhsa-runtime64.so'].hsa_ext_image_data_get_info
    hsa_ext_image_data_get_info.restype = hsa_status_t
    hsa_ext_image_data_get_info.argtypes = [hsa_agent_t, ctypes.POINTER(struct_hsa_ext_image_descriptor_s), hsa_access_permission_t, ctypes.POINTER(struct_hsa_ext_image_data_info_s)]
except AttributeError:
    pass
try:
    hsa_ext_image_data_get_info_with_layout = _libraries['libhsa-runtime64.so'].hsa_ext_image_data_get_info_with_layout
    hsa_ext_image_data_get_info_with_layout.restype = hsa_status_t
    hsa_ext_image_data_get_info_with_layout.argtypes = [hsa_agent_t, ctypes.POINTER(struct_hsa_ext_image_descriptor_s), hsa_access_permission_t, hsa_ext_image_data_layout_t, size_t, size_t, ctypes.POINTER(struct_hsa_ext_image_data_info_s)]
except AttributeError:
    pass
try:
    hsa_ext_image_create = _libraries['libhsa-runtime64.so'].hsa_ext_image_create
    hsa_ext_image_create.restype = hsa_status_t
    hsa_ext_image_create.argtypes = [hsa_agent_t, ctypes.POINTER(struct_hsa_ext_image_descriptor_s), ctypes.POINTER(None), hsa_access_permission_t, ctypes.POINTER(struct_hsa_ext_image_s)]
except AttributeError:
    pass
try:
    hsa_ext_image_create_with_layout = _libraries['libhsa-runtime64.so'].hsa_ext_image_create_with_layout
    hsa_ext_image_create_with_layout.restype = hsa_status_t
    hsa_ext_image_create_with_layout.argtypes = [hsa_agent_t, ctypes.POINTER(struct_hsa_ext_image_descriptor_s), ctypes.POINTER(None), hsa_access_permission_t, hsa_ext_image_data_layout_t, size_t, size_t, ctypes.POINTER(struct_hsa_ext_image_s)]
except AttributeError:
    pass
try:
    hsa_ext_image_destroy = _libraries['libhsa-runtime64.so'].hsa_ext_image_destroy
    hsa_ext_image_destroy.restype = hsa_status_t
    hsa_ext_image_destroy.argtypes = [hsa_agent_t, hsa_ext_image_t]
except AttributeError:
    pass
try:
    hsa_ext_image_copy = _libraries['libhsa-runtime64.so'].hsa_ext_image_copy
    hsa_ext_image_copy.restype = hsa_status_t
    hsa_ext_image_copy.argtypes = [hsa_agent_t, hsa_ext_image_t, ctypes.POINTER(struct_hsa_dim3_s), hsa_ext_image_t, ctypes.POINTER(struct_hsa_dim3_s), ctypes.POINTER(struct_hsa_dim3_s)]
except AttributeError:
    pass
class struct_hsa_ext_image_region_s(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('offset', hsa_dim3_t),
    ('range', hsa_dim3_t),
     ]

hsa_ext_image_region_t = struct_hsa_ext_image_region_s
try:
    hsa_ext_image_import = _libraries['libhsa-runtime64.so'].hsa_ext_image_import
    hsa_ext_image_import.restype = hsa_status_t
    hsa_ext_image_import.argtypes = [hsa_agent_t, ctypes.POINTER(None), size_t, size_t, hsa_ext_image_t, ctypes.POINTER(struct_hsa_ext_image_region_s)]
except AttributeError:
    pass
try:
    hsa_ext_image_export = _libraries['libhsa-runtime64.so'].hsa_ext_image_export
    hsa_ext_image_export.restype = hsa_status_t
    hsa_ext_image_export.argtypes = [hsa_agent_t, hsa_ext_image_t, ctypes.POINTER(None), size_t, size_t, ctypes.POINTER(struct_hsa_ext_image_region_s)]
except AttributeError:
    pass
try:
    hsa_ext_image_clear = _libraries['libhsa-runtime64.so'].hsa_ext_image_clear
    hsa_ext_image_clear.restype = hsa_status_t
    hsa_ext_image_clear.argtypes = [hsa_agent_t, hsa_ext_image_t, ctypes.POINTER(None), ctypes.POINTER(struct_hsa_ext_image_region_s)]
except AttributeError:
    pass
class struct_hsa_ext_sampler_s(Structure):
    pass

struct_hsa_ext_sampler_s._pack_ = 1 # source:False
struct_hsa_ext_sampler_s._fields_ = [
    ('handle', ctypes.c_uint64),
]

hsa_ext_sampler_t = struct_hsa_ext_sampler_s

# values for enumeration 'hsa_ext_sampler_addressing_mode_t'
hsa_ext_sampler_addressing_mode_t__enumvalues = {
    0: 'HSA_EXT_SAMPLER_ADDRESSING_MODE_UNDEFINED',
    1: 'HSA_EXT_SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE',
    2: 'HSA_EXT_SAMPLER_ADDRESSING_MODE_CLAMP_TO_BORDER',
    3: 'HSA_EXT_SAMPLER_ADDRESSING_MODE_REPEAT',
    4: 'HSA_EXT_SAMPLER_ADDRESSING_MODE_MIRRORED_REPEAT',
}
HSA_EXT_SAMPLER_ADDRESSING_MODE_UNDEFINED = 0
HSA_EXT_SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE = 1
HSA_EXT_SAMPLER_ADDRESSING_MODE_CLAMP_TO_BORDER = 2
HSA_EXT_SAMPLER_ADDRESSING_MODE_REPEAT = 3
HSA_EXT_SAMPLER_ADDRESSING_MODE_MIRRORED_REPEAT = 4
hsa_ext_sampler_addressing_mode_t = ctypes.c_uint32 # enum
hsa_ext_sampler_addressing_mode32_t = ctypes.c_uint32

# values for enumeration 'hsa_ext_sampler_coordinate_mode_t'
hsa_ext_sampler_coordinate_mode_t__enumvalues = {
    0: 'HSA_EXT_SAMPLER_COORDINATE_MODE_UNNORMALIZED',
    1: 'HSA_EXT_SAMPLER_COORDINATE_MODE_NORMALIZED',
}
HSA_EXT_SAMPLER_COORDINATE_MODE_UNNORMALIZED = 0
HSA_EXT_SAMPLER_COORDINATE_MODE_NORMALIZED = 1
hsa_ext_sampler_coordinate_mode_t = ctypes.c_uint32 # enum
hsa_ext_sampler_coordinate_mode32_t = ctypes.c_uint32

# values for enumeration 'hsa_ext_sampler_filter_mode_t'
hsa_ext_sampler_filter_mode_t__enumvalues = {
    0: 'HSA_EXT_SAMPLER_FILTER_MODE_NEAREST',
    1: 'HSA_EXT_SAMPLER_FILTER_MODE_LINEAR',
}
HSA_EXT_SAMPLER_FILTER_MODE_NEAREST = 0
HSA_EXT_SAMPLER_FILTER_MODE_LINEAR = 1
hsa_ext_sampler_filter_mode_t = ctypes.c_uint32 # enum
hsa_ext_sampler_filter_mode32_t = ctypes.c_uint32
class struct_hsa_ext_sampler_descriptor_s(Structure):
    pass

struct_hsa_ext_sampler_descriptor_s._pack_ = 1 # source:False
struct_hsa_ext_sampler_descriptor_s._fields_ = [
    ('coordinate_mode', ctypes.c_uint32),
    ('filter_mode', ctypes.c_uint32),
    ('address_mode', ctypes.c_uint32),
]

hsa_ext_sampler_descriptor_t = struct_hsa_ext_sampler_descriptor_s
try:
    hsa_ext_sampler_create = _libraries['libhsa-runtime64.so'].hsa_ext_sampler_create
    hsa_ext_sampler_create.restype = hsa_status_t
    hsa_ext_sampler_create.argtypes = [hsa_agent_t, ctypes.POINTER(struct_hsa_ext_sampler_descriptor_s), ctypes.POINTER(struct_hsa_ext_sampler_s)]
except AttributeError:
    pass
try:
    hsa_ext_sampler_destroy = _libraries['libhsa-runtime64.so'].hsa_ext_sampler_destroy
    hsa_ext_sampler_destroy.restype = hsa_status_t
    hsa_ext_sampler_destroy.argtypes = [hsa_agent_t, hsa_ext_sampler_t]
except AttributeError:
    pass
class struct_hsa_ext_images_1_00_pfn_s(Structure):
    pass

struct_hsa_ext_images_1_00_pfn_s._pack_ = 1 # source:False
struct_hsa_ext_images_1_00_pfn_s._fields_ = [
    ('hsa_ext_image_get_capability', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_agent_s, hsa_ext_image_geometry_t, ctypes.POINTER(struct_hsa_ext_image_format_s), ctypes.POINTER(ctypes.c_uint32))),
    ('hsa_ext_image_data_get_info', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_agent_s, ctypes.POINTER(struct_hsa_ext_image_descriptor_s), hsa_access_permission_t, ctypes.POINTER(struct_hsa_ext_image_data_info_s))),
    ('hsa_ext_image_create', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_agent_s, ctypes.POINTER(struct_hsa_ext_image_descriptor_s), ctypes.POINTER(None), hsa_access_permission_t, ctypes.POINTER(struct_hsa_ext_image_s))),
    ('hsa_ext_image_destroy', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_agent_s, struct_hsa_ext_image_s)),
    ('hsa_ext_image_copy', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_agent_s, struct_hsa_ext_image_s, ctypes.POINTER(struct_hsa_dim3_s), struct_hsa_ext_image_s, ctypes.POINTER(struct_hsa_dim3_s), ctypes.POINTER(struct_hsa_dim3_s))),
    ('hsa_ext_image_import', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_agent_s, ctypes.POINTER(None), ctypes.c_uint64, ctypes.c_uint64, struct_hsa_ext_image_s, ctypes.POINTER(struct_hsa_ext_image_region_s))),
    ('hsa_ext_image_export', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_agent_s, struct_hsa_ext_image_s, ctypes.POINTER(None), ctypes.c_uint64, ctypes.c_uint64, ctypes.POINTER(struct_hsa_ext_image_region_s))),
    ('hsa_ext_image_clear', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_agent_s, struct_hsa_ext_image_s, ctypes.POINTER(None), ctypes.POINTER(struct_hsa_ext_image_region_s))),
    ('hsa_ext_sampler_create', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_agent_s, ctypes.POINTER(struct_hsa_ext_sampler_descriptor_s), ctypes.POINTER(struct_hsa_ext_sampler_s))),
    ('hsa_ext_sampler_destroy', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_agent_s, struct_hsa_ext_sampler_s)),
]

hsa_ext_images_1_00_pfn_t = struct_hsa_ext_images_1_00_pfn_s
class struct_hsa_ext_images_1_pfn_s(Structure):
    pass

struct_hsa_ext_images_1_pfn_s._pack_ = 1 # source:False
struct_hsa_ext_images_1_pfn_s._fields_ = [
    ('hsa_ext_image_get_capability', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_agent_s, hsa_ext_image_geometry_t, ctypes.POINTER(struct_hsa_ext_image_format_s), ctypes.POINTER(ctypes.c_uint32))),
    ('hsa_ext_image_data_get_info', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_agent_s, ctypes.POINTER(struct_hsa_ext_image_descriptor_s), hsa_access_permission_t, ctypes.POINTER(struct_hsa_ext_image_data_info_s))),
    ('hsa_ext_image_create', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_agent_s, ctypes.POINTER(struct_hsa_ext_image_descriptor_s), ctypes.POINTER(None), hsa_access_permission_t, ctypes.POINTER(struct_hsa_ext_image_s))),
    ('hsa_ext_image_destroy', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_agent_s, struct_hsa_ext_image_s)),
    ('hsa_ext_image_copy', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_agent_s, struct_hsa_ext_image_s, ctypes.POINTER(struct_hsa_dim3_s), struct_hsa_ext_image_s, ctypes.POINTER(struct_hsa_dim3_s), ctypes.POINTER(struct_hsa_dim3_s))),
    ('hsa_ext_image_import', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_agent_s, ctypes.POINTER(None), ctypes.c_uint64, ctypes.c_uint64, struct_hsa_ext_image_s, ctypes.POINTER(struct_hsa_ext_image_region_s))),
    ('hsa_ext_image_export', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_agent_s, struct_hsa_ext_image_s, ctypes.POINTER(None), ctypes.c_uint64, ctypes.c_uint64, ctypes.POINTER(struct_hsa_ext_image_region_s))),
    ('hsa_ext_image_clear', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_agent_s, struct_hsa_ext_image_s, ctypes.POINTER(None), ctypes.POINTER(struct_hsa_ext_image_region_s))),
    ('hsa_ext_sampler_create', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_agent_s, ctypes.POINTER(struct_hsa_ext_sampler_descriptor_s), ctypes.POINTER(struct_hsa_ext_sampler_s))),
    ('hsa_ext_sampler_destroy', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_agent_s, struct_hsa_ext_sampler_s)),
    ('hsa_ext_image_get_capability_with_layout', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_agent_s, hsa_ext_image_geometry_t, ctypes.POINTER(struct_hsa_ext_image_format_s), hsa_ext_image_data_layout_t, ctypes.POINTER(ctypes.c_uint32))),
    ('hsa_ext_image_data_get_info_with_layout', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_agent_s, ctypes.POINTER(struct_hsa_ext_image_descriptor_s), hsa_access_permission_t, hsa_ext_image_data_layout_t, ctypes.c_uint64, ctypes.c_uint64, ctypes.POINTER(struct_hsa_ext_image_data_info_s))),
    ('hsa_ext_image_create_with_layout', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_agent_s, ctypes.POINTER(struct_hsa_ext_image_descriptor_s), ctypes.POINTER(None), hsa_access_permission_t, hsa_ext_image_data_layout_t, ctypes.c_uint64, ctypes.c_uint64, ctypes.POINTER(struct_hsa_ext_image_s))),
]

hsa_ext_images_1_pfn_t = struct_hsa_ext_images_1_pfn_s
hsa_signal_condition32_t = ctypes.c_uint32

# values for enumeration 'hsa_amd_packet_type_t'
hsa_amd_packet_type_t__enumvalues = {
    2: 'HSA_AMD_PACKET_TYPE_BARRIER_VALUE',
}
HSA_AMD_PACKET_TYPE_BARRIER_VALUE = 2
hsa_amd_packet_type_t = ctypes.c_uint32 # enum
hsa_amd_packet_type8_t = ctypes.c_ubyte
class struct_hsa_amd_packet_header_s(Structure):
    pass

struct_hsa_amd_packet_header_s._pack_ = 1 # source:False
struct_hsa_amd_packet_header_s._fields_ = [
    ('header', ctypes.c_uint16),
    ('AmdFormat', ctypes.c_ubyte),
    ('reserved', ctypes.c_ubyte),
]

hsa_amd_vendor_packet_header_t = struct_hsa_amd_packet_header_s
class struct_hsa_amd_barrier_value_packet_s(Structure):
    pass

struct_hsa_amd_barrier_value_packet_s._pack_ = 1 # source:False
struct_hsa_amd_barrier_value_packet_s._fields_ = [
    ('header', hsa_amd_vendor_packet_header_t),
    ('reserved0', ctypes.c_uint32),
    ('signal', hsa_signal_t),
    ('value', ctypes.c_int64),
    ('mask', ctypes.c_int64),
    ('cond', ctypes.c_uint32),
    ('reserved1', ctypes.c_uint32),
    ('reserved2', ctypes.c_uint64),
    ('reserved3', ctypes.c_uint64),
    ('completion_signal', hsa_signal_t),
]

hsa_amd_barrier_value_packet_t = struct_hsa_amd_barrier_value_packet_s

# values for enumeration 'enum_hsa_ext_amd_h_179'
enum_hsa_ext_amd_h_179__enumvalues = {
    40: 'HSA_STATUS_ERROR_INVALID_MEMORY_POOL',
    41: 'HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION',
    42: 'HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION',
    43: 'HSA_STATUS_ERROR_MEMORY_FAULT',
    44: 'HSA_STATUS_CU_MASK_REDUCED',
    45: 'HSA_STATUS_ERROR_OUT_OF_REGISTERS',
}
HSA_STATUS_ERROR_INVALID_MEMORY_POOL = 40
HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION = 41
HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION = 42
HSA_STATUS_ERROR_MEMORY_FAULT = 43
HSA_STATUS_CU_MASK_REDUCED = 44
HSA_STATUS_ERROR_OUT_OF_REGISTERS = 45
enum_hsa_ext_amd_h_179 = ctypes.c_uint32 # enum

# values for enumeration 'hsa_amd_iommu_version_t'
hsa_amd_iommu_version_t__enumvalues = {
    0: 'HSA_IOMMU_SUPPORT_NONE',
    1: 'HSA_IOMMU_SUPPORT_V2',
}
HSA_IOMMU_SUPPORT_NONE = 0
HSA_IOMMU_SUPPORT_V2 = 1
hsa_amd_iommu_version_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_amd_agent_info_s'
hsa_amd_agent_info_s__enumvalues = {
    40960: 'HSA_AMD_AGENT_INFO_CHIP_ID',
    40961: 'HSA_AMD_AGENT_INFO_CACHELINE_SIZE',
    40962: 'HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT',
    40963: 'HSA_AMD_AGENT_INFO_MAX_CLOCK_FREQUENCY',
    40964: 'HSA_AMD_AGENT_INFO_DRIVER_NODE_ID',
    40965: 'HSA_AMD_AGENT_INFO_MAX_ADDRESS_WATCH_POINTS',
    40966: 'HSA_AMD_AGENT_INFO_BDFID',
    40967: 'HSA_AMD_AGENT_INFO_MEMORY_WIDTH',
    40968: 'HSA_AMD_AGENT_INFO_MEMORY_MAX_FREQUENCY',
    40969: 'HSA_AMD_AGENT_INFO_PRODUCT_NAME',
    40970: 'HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU',
    40971: 'HSA_AMD_AGENT_INFO_NUM_SIMDS_PER_CU',
    40972: 'HSA_AMD_AGENT_INFO_NUM_SHADER_ENGINES',
    40973: 'HSA_AMD_AGENT_INFO_NUM_SHADER_ARRAYS_PER_SE',
    40974: 'HSA_AMD_AGENT_INFO_HDP_FLUSH',
    40975: 'HSA_AMD_AGENT_INFO_DOMAIN',
    40976: 'HSA_AMD_AGENT_INFO_COOPERATIVE_QUEUES',
    40977: 'HSA_AMD_AGENT_INFO_UUID',
    40978: 'HSA_AMD_AGENT_INFO_ASIC_REVISION',
    40979: 'HSA_AMD_AGENT_INFO_SVM_DIRECT_HOST_ACCESS',
    40980: 'HSA_AMD_AGENT_INFO_COOPERATIVE_COMPUTE_UNIT_COUNT',
    40981: 'HSA_AMD_AGENT_INFO_MEMORY_AVAIL',
    40982: 'HSA_AMD_AGENT_INFO_TIMESTAMP_FREQUENCY',
    41223: 'HSA_AMD_AGENT_INFO_ASIC_FAMILY_ID',
    41224: 'HSA_AMD_AGENT_INFO_UCODE_VERSION',
    41225: 'HSA_AMD_AGENT_INFO_SDMA_UCODE_VERSION',
    41226: 'HSA_AMD_AGENT_INFO_NUM_SDMA_ENG',
    41227: 'HSA_AMD_AGENT_INFO_NUM_SDMA_XGMI_ENG',
    41232: 'HSA_AMD_AGENT_INFO_IOMMU_SUPPORT',
    41233: 'HSA_AMD_AGENT_INFO_NUM_XCC',
    41234: 'HSA_AMD_AGENT_INFO_DRIVER_UID',
    41235: 'HSA_AMD_AGENT_INFO_NEAREST_CPU',
}
HSA_AMD_AGENT_INFO_CHIP_ID = 40960
HSA_AMD_AGENT_INFO_CACHELINE_SIZE = 40961
HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT = 40962
HSA_AMD_AGENT_INFO_MAX_CLOCK_FREQUENCY = 40963
HSA_AMD_AGENT_INFO_DRIVER_NODE_ID = 40964
HSA_AMD_AGENT_INFO_MAX_ADDRESS_WATCH_POINTS = 40965
HSA_AMD_AGENT_INFO_BDFID = 40966
HSA_AMD_AGENT_INFO_MEMORY_WIDTH = 40967
HSA_AMD_AGENT_INFO_MEMORY_MAX_FREQUENCY = 40968
HSA_AMD_AGENT_INFO_PRODUCT_NAME = 40969
HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU = 40970
HSA_AMD_AGENT_INFO_NUM_SIMDS_PER_CU = 40971
HSA_AMD_AGENT_INFO_NUM_SHADER_ENGINES = 40972
HSA_AMD_AGENT_INFO_NUM_SHADER_ARRAYS_PER_SE = 40973
HSA_AMD_AGENT_INFO_HDP_FLUSH = 40974
HSA_AMD_AGENT_INFO_DOMAIN = 40975
HSA_AMD_AGENT_INFO_COOPERATIVE_QUEUES = 40976
HSA_AMD_AGENT_INFO_UUID = 40977
HSA_AMD_AGENT_INFO_ASIC_REVISION = 40978
HSA_AMD_AGENT_INFO_SVM_DIRECT_HOST_ACCESS = 40979
HSA_AMD_AGENT_INFO_COOPERATIVE_COMPUTE_UNIT_COUNT = 40980
HSA_AMD_AGENT_INFO_MEMORY_AVAIL = 40981
HSA_AMD_AGENT_INFO_TIMESTAMP_FREQUENCY = 40982
HSA_AMD_AGENT_INFO_ASIC_FAMILY_ID = 41223
HSA_AMD_AGENT_INFO_UCODE_VERSION = 41224
HSA_AMD_AGENT_INFO_SDMA_UCODE_VERSION = 41225
HSA_AMD_AGENT_INFO_NUM_SDMA_ENG = 41226
HSA_AMD_AGENT_INFO_NUM_SDMA_XGMI_ENG = 41227
HSA_AMD_AGENT_INFO_IOMMU_SUPPORT = 41232
HSA_AMD_AGENT_INFO_NUM_XCC = 41233
HSA_AMD_AGENT_INFO_DRIVER_UID = 41234
HSA_AMD_AGENT_INFO_NEAREST_CPU = 41235
hsa_amd_agent_info_s = ctypes.c_uint32 # enum
hsa_amd_agent_info_t = hsa_amd_agent_info_s
hsa_amd_agent_info_t__enumvalues = hsa_amd_agent_info_s__enumvalues

# values for enumeration 'hsa_amd_sdma_engine_id'
hsa_amd_sdma_engine_id__enumvalues = {
    1: 'HSA_AMD_SDMA_ENGINE_0',
    2: 'HSA_AMD_SDMA_ENGINE_1',
    4: 'HSA_AMD_SDMA_ENGINE_2',
    8: 'HSA_AMD_SDMA_ENGINE_3',
    16: 'HSA_AMD_SDMA_ENGINE_4',
    32: 'HSA_AMD_SDMA_ENGINE_5',
    64: 'HSA_AMD_SDMA_ENGINE_6',
    128: 'HSA_AMD_SDMA_ENGINE_7',
    256: 'HSA_AMD_SDMA_ENGINE_8',
    512: 'HSA_AMD_SDMA_ENGINE_9',
    1024: 'HSA_AMD_SDMA_ENGINE_10',
    2048: 'HSA_AMD_SDMA_ENGINE_11',
    4096: 'HSA_AMD_SDMA_ENGINE_12',
    8192: 'HSA_AMD_SDMA_ENGINE_13',
    16384: 'HSA_AMD_SDMA_ENGINE_14',
    32768: 'HSA_AMD_SDMA_ENGINE_15',
}
HSA_AMD_SDMA_ENGINE_0 = 1
HSA_AMD_SDMA_ENGINE_1 = 2
HSA_AMD_SDMA_ENGINE_2 = 4
HSA_AMD_SDMA_ENGINE_3 = 8
HSA_AMD_SDMA_ENGINE_4 = 16
HSA_AMD_SDMA_ENGINE_5 = 32
HSA_AMD_SDMA_ENGINE_6 = 64
HSA_AMD_SDMA_ENGINE_7 = 128
HSA_AMD_SDMA_ENGINE_8 = 256
HSA_AMD_SDMA_ENGINE_9 = 512
HSA_AMD_SDMA_ENGINE_10 = 1024
HSA_AMD_SDMA_ENGINE_11 = 2048
HSA_AMD_SDMA_ENGINE_12 = 4096
HSA_AMD_SDMA_ENGINE_13 = 8192
HSA_AMD_SDMA_ENGINE_14 = 16384
HSA_AMD_SDMA_ENGINE_15 = 32768
hsa_amd_sdma_engine_id = ctypes.c_uint32 # enum
hsa_amd_sdma_engine_id_t = hsa_amd_sdma_engine_id
hsa_amd_sdma_engine_id_t__enumvalues = hsa_amd_sdma_engine_id__enumvalues
class struct_hsa_amd_hdp_flush_s(Structure):
    pass

struct_hsa_amd_hdp_flush_s._pack_ = 1 # source:False
struct_hsa_amd_hdp_flush_s._fields_ = [
    ('HDP_MEM_FLUSH_CNTL', ctypes.POINTER(ctypes.c_uint32)),
    ('HDP_REG_FLUSH_CNTL', ctypes.POINTER(ctypes.c_uint32)),
]

hsa_amd_hdp_flush_t = struct_hsa_amd_hdp_flush_s

# values for enumeration 'hsa_amd_region_info_s'
hsa_amd_region_info_s__enumvalues = {
    40960: 'HSA_AMD_REGION_INFO_HOST_ACCESSIBLE',
    40961: 'HSA_AMD_REGION_INFO_BASE',
    40962: 'HSA_AMD_REGION_INFO_BUS_WIDTH',
    40963: 'HSA_AMD_REGION_INFO_MAX_CLOCK_FREQUENCY',
}
HSA_AMD_REGION_INFO_HOST_ACCESSIBLE = 40960
HSA_AMD_REGION_INFO_BASE = 40961
HSA_AMD_REGION_INFO_BUS_WIDTH = 40962
HSA_AMD_REGION_INFO_MAX_CLOCK_FREQUENCY = 40963
hsa_amd_region_info_s = ctypes.c_uint32 # enum
hsa_amd_region_info_t = hsa_amd_region_info_s
hsa_amd_region_info_t__enumvalues = hsa_amd_region_info_s__enumvalues

# values for enumeration 'hsa_amd_coherency_type_s'
hsa_amd_coherency_type_s__enumvalues = {
    0: 'HSA_AMD_COHERENCY_TYPE_COHERENT',
    1: 'HSA_AMD_COHERENCY_TYPE_NONCOHERENT',
}
HSA_AMD_COHERENCY_TYPE_COHERENT = 0
HSA_AMD_COHERENCY_TYPE_NONCOHERENT = 1
hsa_amd_coherency_type_s = ctypes.c_uint32 # enum
hsa_amd_coherency_type_t = hsa_amd_coherency_type_s
hsa_amd_coherency_type_t__enumvalues = hsa_amd_coherency_type_s__enumvalues
try:
    hsa_amd_coherency_get_type = _libraries['libhsa-runtime64.so'].hsa_amd_coherency_get_type
    hsa_amd_coherency_get_type.restype = hsa_status_t
    hsa_amd_coherency_get_type.argtypes = [hsa_agent_t, ctypes.POINTER(hsa_amd_coherency_type_s)]
except AttributeError:
    pass
try:
    hsa_amd_coherency_set_type = _libraries['libhsa-runtime64.so'].hsa_amd_coherency_set_type
    hsa_amd_coherency_set_type.restype = hsa_status_t
    hsa_amd_coherency_set_type.argtypes = [hsa_agent_t, hsa_amd_coherency_type_t]
except AttributeError:
    pass
class struct_hsa_amd_profiling_dispatch_time_s(Structure):
    pass

struct_hsa_amd_profiling_dispatch_time_s._pack_ = 1 # source:False
struct_hsa_amd_profiling_dispatch_time_s._fields_ = [
    ('start', ctypes.c_uint64),
    ('end', ctypes.c_uint64),
]

hsa_amd_profiling_dispatch_time_t = struct_hsa_amd_profiling_dispatch_time_s
class struct_hsa_amd_profiling_async_copy_time_s(Structure):
    pass

struct_hsa_amd_profiling_async_copy_time_s._pack_ = 1 # source:False
struct_hsa_amd_profiling_async_copy_time_s._fields_ = [
    ('start', ctypes.c_uint64),
    ('end', ctypes.c_uint64),
]

hsa_amd_profiling_async_copy_time_t = struct_hsa_amd_profiling_async_copy_time_s
try:
    hsa_amd_profiling_set_profiler_enabled = _libraries['libhsa-runtime64.so'].hsa_amd_profiling_set_profiler_enabled
    hsa_amd_profiling_set_profiler_enabled.restype = hsa_status_t
    hsa_amd_profiling_set_profiler_enabled.argtypes = [ctypes.POINTER(struct_hsa_queue_s), ctypes.c_int32]
except AttributeError:
    pass
try:
    hsa_amd_profiling_async_copy_enable = _libraries['libhsa-runtime64.so'].hsa_amd_profiling_async_copy_enable
    hsa_amd_profiling_async_copy_enable.restype = hsa_status_t
    hsa_amd_profiling_async_copy_enable.argtypes = [ctypes.c_bool]
except AttributeError:
    pass
try:
    hsa_amd_profiling_get_dispatch_time = _libraries['libhsa-runtime64.so'].hsa_amd_profiling_get_dispatch_time
    hsa_amd_profiling_get_dispatch_time.restype = hsa_status_t
    hsa_amd_profiling_get_dispatch_time.argtypes = [hsa_agent_t, hsa_signal_t, ctypes.POINTER(struct_hsa_amd_profiling_dispatch_time_s)]
except AttributeError:
    pass
try:
    hsa_amd_profiling_get_async_copy_time = _libraries['libhsa-runtime64.so'].hsa_amd_profiling_get_async_copy_time
    hsa_amd_profiling_get_async_copy_time.restype = hsa_status_t
    hsa_amd_profiling_get_async_copy_time.argtypes = [hsa_signal_t, ctypes.POINTER(struct_hsa_amd_profiling_async_copy_time_s)]
except AttributeError:
    pass
try:
    hsa_amd_profiling_convert_tick_to_system_domain = _libraries['libhsa-runtime64.so'].hsa_amd_profiling_convert_tick_to_system_domain
    hsa_amd_profiling_convert_tick_to_system_domain.restype = hsa_status_t
    hsa_amd_profiling_convert_tick_to_system_domain.argtypes = [hsa_agent_t, uint64_t, ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass

# values for enumeration 'hsa_amd_signal_attribute_t'
hsa_amd_signal_attribute_t__enumvalues = {
    1: 'HSA_AMD_SIGNAL_AMD_GPU_ONLY',
    2: 'HSA_AMD_SIGNAL_IPC',
}
HSA_AMD_SIGNAL_AMD_GPU_ONLY = 1
HSA_AMD_SIGNAL_IPC = 2
hsa_amd_signal_attribute_t = ctypes.c_uint32 # enum
try:
    hsa_amd_signal_create = _libraries['libhsa-runtime64.so'].hsa_amd_signal_create
    hsa_amd_signal_create.restype = hsa_status_t
    hsa_amd_signal_create.argtypes = [hsa_signal_value_t, uint32_t, ctypes.POINTER(struct_hsa_agent_s), uint64_t, ctypes.POINTER(struct_hsa_signal_s)]
except AttributeError:
    pass
try:
    hsa_amd_signal_value_pointer = _libraries['libhsa-runtime64.so'].hsa_amd_signal_value_pointer
    hsa_amd_signal_value_pointer.restype = hsa_status_t
    hsa_amd_signal_value_pointer.argtypes = [hsa_signal_t, ctypes.POINTER(ctypes.POINTER(ctypes.c_int64))]
except AttributeError:
    pass
hsa_amd_signal_handler = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_int64, ctypes.POINTER(None))
try:
    hsa_amd_signal_async_handler = _libraries['libhsa-runtime64.so'].hsa_amd_signal_async_handler
    hsa_amd_signal_async_handler.restype = hsa_status_t
    hsa_amd_signal_async_handler.argtypes = [hsa_signal_t, hsa_signal_condition_t, hsa_signal_value_t, hsa_amd_signal_handler, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_amd_async_function = _libraries['libhsa-runtime64.so'].hsa_amd_async_function
    hsa_amd_async_function.restype = hsa_status_t
    hsa_amd_async_function.argtypes = [ctypes.CFUNCTYPE(None, ctypes.POINTER(None)), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_amd_signal_wait_any = _libraries['libhsa-runtime64.so'].hsa_amd_signal_wait_any
    hsa_amd_signal_wait_any.restype = uint32_t
    hsa_amd_signal_wait_any.argtypes = [uint32_t, ctypes.POINTER(struct_hsa_signal_s), ctypes.POINTER(hsa_signal_condition_t), ctypes.POINTER(ctypes.c_int64), uint64_t, hsa_wait_state_t, ctypes.POINTER(ctypes.c_int64)]
except AttributeError:
    pass
try:
    hsa_amd_image_get_info_max_dim = _libraries['libhsa-runtime64.so'].hsa_amd_image_get_info_max_dim
    hsa_amd_image_get_info_max_dim.restype = hsa_status_t
    hsa_amd_image_get_info_max_dim.argtypes = [hsa_agent_t, hsa_agent_info_t, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_amd_queue_cu_set_mask = _libraries['libhsa-runtime64.so'].hsa_amd_queue_cu_set_mask
    hsa_amd_queue_cu_set_mask.restype = hsa_status_t
    hsa_amd_queue_cu_set_mask.argtypes = [ctypes.POINTER(struct_hsa_queue_s), uint32_t, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
try:
    hsa_amd_queue_cu_get_mask = _libraries['libhsa-runtime64.so'].hsa_amd_queue_cu_get_mask
    hsa_amd_queue_cu_get_mask.restype = hsa_status_t
    hsa_amd_queue_cu_get_mask.argtypes = [ctypes.POINTER(struct_hsa_queue_s), uint32_t, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass

# values for enumeration 'hsa_amd_segment_t'
hsa_amd_segment_t__enumvalues = {
    0: 'HSA_AMD_SEGMENT_GLOBAL',
    1: 'HSA_AMD_SEGMENT_READONLY',
    2: 'HSA_AMD_SEGMENT_PRIVATE',
    3: 'HSA_AMD_SEGMENT_GROUP',
}
HSA_AMD_SEGMENT_GLOBAL = 0
HSA_AMD_SEGMENT_READONLY = 1
HSA_AMD_SEGMENT_PRIVATE = 2
HSA_AMD_SEGMENT_GROUP = 3
hsa_amd_segment_t = ctypes.c_uint32 # enum
class struct_hsa_amd_memory_pool_s(Structure):
    pass

struct_hsa_amd_memory_pool_s._pack_ = 1 # source:False
struct_hsa_amd_memory_pool_s._fields_ = [
    ('handle', ctypes.c_uint64),
]

hsa_amd_memory_pool_t = struct_hsa_amd_memory_pool_s

# values for enumeration 'hsa_amd_memory_pool_global_flag_s'
hsa_amd_memory_pool_global_flag_s__enumvalues = {
    1: 'HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT',
    2: 'HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED',
    4: 'HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED',
    8: 'HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED',
}
HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT = 1
HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED = 2
HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED = 4
HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED = 8
hsa_amd_memory_pool_global_flag_s = ctypes.c_uint32 # enum
hsa_amd_memory_pool_global_flag_t = hsa_amd_memory_pool_global_flag_s
hsa_amd_memory_pool_global_flag_t__enumvalues = hsa_amd_memory_pool_global_flag_s__enumvalues

# values for enumeration 'hsa_amd_memory_pool_location_s'
hsa_amd_memory_pool_location_s__enumvalues = {
    0: 'HSA_AMD_MEMORY_POOL_LOCATION_CPU',
    1: 'HSA_AMD_MEMORY_POOL_LOCATION_GPU',
}
HSA_AMD_MEMORY_POOL_LOCATION_CPU = 0
HSA_AMD_MEMORY_POOL_LOCATION_GPU = 1
hsa_amd_memory_pool_location_s = ctypes.c_uint32 # enum
hsa_amd_memory_pool_location_t = hsa_amd_memory_pool_location_s
hsa_amd_memory_pool_location_t__enumvalues = hsa_amd_memory_pool_location_s__enumvalues

# values for enumeration 'hsa_amd_memory_pool_info_t'
hsa_amd_memory_pool_info_t__enumvalues = {
    0: 'HSA_AMD_MEMORY_POOL_INFO_SEGMENT',
    1: 'HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS',
    2: 'HSA_AMD_MEMORY_POOL_INFO_SIZE',
    5: 'HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED',
    6: 'HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE',
    7: 'HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALIGNMENT',
    15: 'HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL',
    16: 'HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE',
    17: 'HSA_AMD_MEMORY_POOL_INFO_LOCATION',
    18: 'HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE',
}
HSA_AMD_MEMORY_POOL_INFO_SEGMENT = 0
HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS = 1
HSA_AMD_MEMORY_POOL_INFO_SIZE = 2
HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED = 5
HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE = 6
HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALIGNMENT = 7
HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL = 15
HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE = 16
HSA_AMD_MEMORY_POOL_INFO_LOCATION = 17
HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE = 18
hsa_amd_memory_pool_info_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_amd_memory_pool_flag_s'
hsa_amd_memory_pool_flag_s__enumvalues = {
    0: 'HSA_AMD_MEMORY_POOL_STANDARD_FLAG',
    1: 'HSA_AMD_MEMORY_POOL_PCIE_FLAG',
}
HSA_AMD_MEMORY_POOL_STANDARD_FLAG = 0
HSA_AMD_MEMORY_POOL_PCIE_FLAG = 1
hsa_amd_memory_pool_flag_s = ctypes.c_uint32 # enum
hsa_amd_memory_pool_flag_t = hsa_amd_memory_pool_flag_s
hsa_amd_memory_pool_flag_t__enumvalues = hsa_amd_memory_pool_flag_s__enumvalues
try:
    hsa_amd_memory_pool_get_info = _libraries['libhsa-runtime64.so'].hsa_amd_memory_pool_get_info
    hsa_amd_memory_pool_get_info.restype = hsa_status_t
    hsa_amd_memory_pool_get_info.argtypes = [hsa_amd_memory_pool_t, hsa_amd_memory_pool_info_t, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_amd_agent_iterate_memory_pools = _libraries['libhsa-runtime64.so'].hsa_amd_agent_iterate_memory_pools
    hsa_amd_agent_iterate_memory_pools.restype = hsa_status_t
    hsa_amd_agent_iterate_memory_pools.argtypes = [hsa_agent_t, ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_amd_memory_pool_s, ctypes.POINTER(None)), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_amd_memory_pool_allocate = _libraries['libhsa-runtime64.so'].hsa_amd_memory_pool_allocate
    hsa_amd_memory_pool_allocate.restype = hsa_status_t
    hsa_amd_memory_pool_allocate.argtypes = [hsa_amd_memory_pool_t, size_t, uint32_t, ctypes.POINTER(ctypes.POINTER(None))]
except AttributeError:
    pass
try:
    hsa_amd_memory_pool_free = _libraries['libhsa-runtime64.so'].hsa_amd_memory_pool_free
    hsa_amd_memory_pool_free.restype = hsa_status_t
    hsa_amd_memory_pool_free.argtypes = [ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_amd_memory_async_copy = _libraries['libhsa-runtime64.so'].hsa_amd_memory_async_copy
    hsa_amd_memory_async_copy.restype = hsa_status_t
    hsa_amd_memory_async_copy.argtypes = [ctypes.POINTER(None), hsa_agent_t, ctypes.POINTER(None), hsa_agent_t, size_t, uint32_t, ctypes.POINTER(struct_hsa_signal_s), hsa_signal_t]
except AttributeError:
    pass
try:
    hsa_amd_memory_async_copy_on_engine = _libraries['libhsa-runtime64.so'].hsa_amd_memory_async_copy_on_engine
    hsa_amd_memory_async_copy_on_engine.restype = hsa_status_t
    hsa_amd_memory_async_copy_on_engine.argtypes = [ctypes.POINTER(None), hsa_agent_t, ctypes.POINTER(None), hsa_agent_t, size_t, uint32_t, ctypes.POINTER(struct_hsa_signal_s), hsa_signal_t, hsa_amd_sdma_engine_id_t, ctypes.c_bool]
except AttributeError:
    pass
try:
    hsa_amd_memory_copy_engine_status = _libraries['libhsa-runtime64.so'].hsa_amd_memory_copy_engine_status
    hsa_amd_memory_copy_engine_status.restype = hsa_status_t
    hsa_amd_memory_copy_engine_status.argtypes = [hsa_agent_t, hsa_agent_t, ctypes.POINTER(ctypes.c_uint32)]
except AttributeError:
    pass
class struct_hsa_pitched_ptr_s(Structure):
    pass

struct_hsa_pitched_ptr_s._pack_ = 1 # source:False
struct_hsa_pitched_ptr_s._fields_ = [
    ('base', ctypes.POINTER(None)),
    ('pitch', ctypes.c_uint64),
    ('slice', ctypes.c_uint64),
]

hsa_pitched_ptr_t = struct_hsa_pitched_ptr_s

# values for enumeration 'hsa_amd_copy_direction_t'
hsa_amd_copy_direction_t__enumvalues = {
    0: 'hsaHostToHost',
    1: 'hsaHostToDevice',
    2: 'hsaDeviceToHost',
    3: 'hsaDeviceToDevice',
}
hsaHostToHost = 0
hsaHostToDevice = 1
hsaDeviceToHost = 2
hsaDeviceToDevice = 3
hsa_amd_copy_direction_t = ctypes.c_uint32 # enum
try:
    hsa_amd_memory_async_copy_rect = _libraries['libhsa-runtime64.so'].hsa_amd_memory_async_copy_rect
    hsa_amd_memory_async_copy_rect.restype = hsa_status_t
    hsa_amd_memory_async_copy_rect.argtypes = [ctypes.POINTER(struct_hsa_pitched_ptr_s), ctypes.POINTER(struct_hsa_dim3_s), ctypes.POINTER(struct_hsa_pitched_ptr_s), ctypes.POINTER(struct_hsa_dim3_s), ctypes.POINTER(struct_hsa_dim3_s), hsa_agent_t, hsa_amd_copy_direction_t, uint32_t, ctypes.POINTER(struct_hsa_signal_s), hsa_signal_t]
except AttributeError:
    pass

# values for enumeration 'hsa_amd_memory_pool_access_t'
hsa_amd_memory_pool_access_t__enumvalues = {
    0: 'HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED',
    1: 'HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT',
    2: 'HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT',
}
HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED = 0
HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT = 1
HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT = 2
hsa_amd_memory_pool_access_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_amd_link_info_type_t'
hsa_amd_link_info_type_t__enumvalues = {
    0: 'HSA_AMD_LINK_INFO_TYPE_HYPERTRANSPORT',
    1: 'HSA_AMD_LINK_INFO_TYPE_QPI',
    2: 'HSA_AMD_LINK_INFO_TYPE_PCIE',
    3: 'HSA_AMD_LINK_INFO_TYPE_INFINBAND',
    4: 'HSA_AMD_LINK_INFO_TYPE_XGMI',
}
HSA_AMD_LINK_INFO_TYPE_HYPERTRANSPORT = 0
HSA_AMD_LINK_INFO_TYPE_QPI = 1
HSA_AMD_LINK_INFO_TYPE_PCIE = 2
HSA_AMD_LINK_INFO_TYPE_INFINBAND = 3
HSA_AMD_LINK_INFO_TYPE_XGMI = 4
hsa_amd_link_info_type_t = ctypes.c_uint32 # enum
class struct_hsa_amd_memory_pool_link_info_s(Structure):
    pass

struct_hsa_amd_memory_pool_link_info_s._pack_ = 1 # source:False
struct_hsa_amd_memory_pool_link_info_s._fields_ = [
    ('min_latency', ctypes.c_uint32),
    ('max_latency', ctypes.c_uint32),
    ('min_bandwidth', ctypes.c_uint32),
    ('max_bandwidth', ctypes.c_uint32),
    ('atomic_support_32bit', ctypes.c_bool),
    ('atomic_support_64bit', ctypes.c_bool),
    ('coherent_support', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte),
    ('link_type', hsa_amd_link_info_type_t),
    ('numa_distance', ctypes.c_uint32),
]

hsa_amd_memory_pool_link_info_t = struct_hsa_amd_memory_pool_link_info_s

# values for enumeration 'hsa_amd_agent_memory_pool_info_t'
hsa_amd_agent_memory_pool_info_t__enumvalues = {
    0: 'HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS',
    1: 'HSA_AMD_AGENT_MEMORY_POOL_INFO_NUM_LINK_HOPS',
    2: 'HSA_AMD_AGENT_MEMORY_POOL_INFO_LINK_INFO',
}
HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS = 0
HSA_AMD_AGENT_MEMORY_POOL_INFO_NUM_LINK_HOPS = 1
HSA_AMD_AGENT_MEMORY_POOL_INFO_LINK_INFO = 2
hsa_amd_agent_memory_pool_info_t = ctypes.c_uint32 # enum
try:
    hsa_amd_agent_memory_pool_get_info = _libraries['libhsa-runtime64.so'].hsa_amd_agent_memory_pool_get_info
    hsa_amd_agent_memory_pool_get_info.restype = hsa_status_t
    hsa_amd_agent_memory_pool_get_info.argtypes = [hsa_agent_t, hsa_amd_memory_pool_t, hsa_amd_agent_memory_pool_info_t, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_amd_agents_allow_access = _libraries['libhsa-runtime64.so'].hsa_amd_agents_allow_access
    hsa_amd_agents_allow_access.restype = hsa_status_t
    hsa_amd_agents_allow_access.argtypes = [uint32_t, ctypes.POINTER(struct_hsa_agent_s), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_amd_memory_pool_can_migrate = _libraries['libhsa-runtime64.so'].hsa_amd_memory_pool_can_migrate
    hsa_amd_memory_pool_can_migrate.restype = hsa_status_t
    hsa_amd_memory_pool_can_migrate.argtypes = [hsa_amd_memory_pool_t, hsa_amd_memory_pool_t, ctypes.POINTER(ctypes.c_bool)]
except AttributeError:
    pass
try:
    hsa_amd_memory_migrate = _libraries['libhsa-runtime64.so'].hsa_amd_memory_migrate
    hsa_amd_memory_migrate.restype = hsa_status_t
    hsa_amd_memory_migrate.argtypes = [ctypes.POINTER(None), hsa_amd_memory_pool_t, uint32_t]
except AttributeError:
    pass
try:
    hsa_amd_memory_lock = _libraries['libhsa-runtime64.so'].hsa_amd_memory_lock
    hsa_amd_memory_lock.restype = hsa_status_t
    hsa_amd_memory_lock.argtypes = [ctypes.POINTER(None), size_t, ctypes.POINTER(struct_hsa_agent_s), ctypes.c_int32, ctypes.POINTER(ctypes.POINTER(None))]
except AttributeError:
    pass
try:
    hsa_amd_memory_lock_to_pool = _libraries['libhsa-runtime64.so'].hsa_amd_memory_lock_to_pool
    hsa_amd_memory_lock_to_pool.restype = hsa_status_t
    hsa_amd_memory_lock_to_pool.argtypes = [ctypes.POINTER(None), size_t, ctypes.POINTER(struct_hsa_agent_s), ctypes.c_int32, hsa_amd_memory_pool_t, uint32_t, ctypes.POINTER(ctypes.POINTER(None))]
except AttributeError:
    pass
try:
    hsa_amd_memory_unlock = _libraries['libhsa-runtime64.so'].hsa_amd_memory_unlock
    hsa_amd_memory_unlock.restype = hsa_status_t
    hsa_amd_memory_unlock.argtypes = [ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_amd_memory_fill = _libraries['libhsa-runtime64.so'].hsa_amd_memory_fill
    hsa_amd_memory_fill.restype = hsa_status_t
    hsa_amd_memory_fill.argtypes = [ctypes.POINTER(None), uint32_t, size_t]
except AttributeError:
    pass
try:
    hsa_amd_interop_map_buffer = _libraries['libhsa-runtime64.so'].hsa_amd_interop_map_buffer
    hsa_amd_interop_map_buffer.restype = hsa_status_t
    hsa_amd_interop_map_buffer.argtypes = [uint32_t, ctypes.POINTER(struct_hsa_agent_s), ctypes.c_int32, uint32_t, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.POINTER(None)), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.POINTER(None))]
except AttributeError:
    pass
try:
    hsa_amd_interop_unmap_buffer = _libraries['libhsa-runtime64.so'].hsa_amd_interop_unmap_buffer
    hsa_amd_interop_unmap_buffer.restype = hsa_status_t
    hsa_amd_interop_unmap_buffer.argtypes = [ctypes.POINTER(None)]
except AttributeError:
    pass
class struct_hsa_amd_image_descriptor_s(Structure):
    pass

struct_hsa_amd_image_descriptor_s._pack_ = 1 # source:False
struct_hsa_amd_image_descriptor_s._fields_ = [
    ('version', ctypes.c_uint32),
    ('deviceID', ctypes.c_uint32),
    ('data', ctypes.c_uint32 * 1),
]

hsa_amd_image_descriptor_t = struct_hsa_amd_image_descriptor_s
try:
    hsa_amd_image_create = _libraries['libhsa-runtime64.so'].hsa_amd_image_create
    hsa_amd_image_create.restype = hsa_status_t
    hsa_amd_image_create.argtypes = [hsa_agent_t, ctypes.POINTER(struct_hsa_ext_image_descriptor_s), ctypes.POINTER(struct_hsa_amd_image_descriptor_s), ctypes.POINTER(None), hsa_access_permission_t, ctypes.POINTER(struct_hsa_ext_image_s)]
except AttributeError:
    pass

# values for enumeration 'hsa_amd_pointer_type_t'
hsa_amd_pointer_type_t__enumvalues = {
    0: 'HSA_EXT_POINTER_TYPE_UNKNOWN',
    1: 'HSA_EXT_POINTER_TYPE_HSA',
    2: 'HSA_EXT_POINTER_TYPE_LOCKED',
    3: 'HSA_EXT_POINTER_TYPE_GRAPHICS',
    4: 'HSA_EXT_POINTER_TYPE_IPC',
}
HSA_EXT_POINTER_TYPE_UNKNOWN = 0
HSA_EXT_POINTER_TYPE_HSA = 1
HSA_EXT_POINTER_TYPE_LOCKED = 2
HSA_EXT_POINTER_TYPE_GRAPHICS = 3
HSA_EXT_POINTER_TYPE_IPC = 4
hsa_amd_pointer_type_t = ctypes.c_uint32 # enum
class struct_hsa_amd_pointer_info_s(Structure):
    pass

struct_hsa_amd_pointer_info_s._pack_ = 1 # source:False
struct_hsa_amd_pointer_info_s._fields_ = [
    ('size', ctypes.c_uint32),
    ('type', hsa_amd_pointer_type_t),
    ('agentBaseAddress', ctypes.POINTER(None)),
    ('hostBaseAddress', ctypes.POINTER(None)),
    ('sizeInBytes', ctypes.c_uint64),
    ('userData', ctypes.POINTER(None)),
    ('agentOwner', hsa_agent_t),
    ('global_flags', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

hsa_amd_pointer_info_t = struct_hsa_amd_pointer_info_s
try:
    hsa_amd_pointer_info = _libraries['libhsa-runtime64.so'].hsa_amd_pointer_info
    hsa_amd_pointer_info.restype = hsa_status_t
    hsa_amd_pointer_info.argtypes = [ctypes.POINTER(None), ctypes.POINTER(struct_hsa_amd_pointer_info_s), ctypes.CFUNCTYPE(ctypes.POINTER(None), ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.POINTER(struct_hsa_agent_s))]
except AttributeError:
    pass
try:
    hsa_amd_pointer_info_set_userdata = _libraries['libhsa-runtime64.so'].hsa_amd_pointer_info_set_userdata
    hsa_amd_pointer_info_set_userdata.restype = hsa_status_t
    hsa_amd_pointer_info_set_userdata.argtypes = [ctypes.POINTER(None), ctypes.POINTER(None)]
except AttributeError:
    pass
class struct_hsa_amd_ipc_memory_s(Structure):
    pass

struct_hsa_amd_ipc_memory_s._pack_ = 1 # source:False
struct_hsa_amd_ipc_memory_s._fields_ = [
    ('handle', ctypes.c_uint32 * 8),
]

hsa_amd_ipc_memory_t = struct_hsa_amd_ipc_memory_s
try:
    hsa_amd_ipc_memory_create = _libraries['libhsa-runtime64.so'].hsa_amd_ipc_memory_create
    hsa_amd_ipc_memory_create.restype = hsa_status_t
    hsa_amd_ipc_memory_create.argtypes = [ctypes.POINTER(None), size_t, ctypes.POINTER(struct_hsa_amd_ipc_memory_s)]
except AttributeError:
    pass
try:
    hsa_amd_ipc_memory_attach = _libraries['libhsa-runtime64.so'].hsa_amd_ipc_memory_attach
    hsa_amd_ipc_memory_attach.restype = hsa_status_t
    hsa_amd_ipc_memory_attach.argtypes = [ctypes.POINTER(struct_hsa_amd_ipc_memory_s), size_t, uint32_t, ctypes.POINTER(struct_hsa_agent_s), ctypes.POINTER(ctypes.POINTER(None))]
except AttributeError:
    pass
try:
    hsa_amd_ipc_memory_detach = _libraries['libhsa-runtime64.so'].hsa_amd_ipc_memory_detach
    hsa_amd_ipc_memory_detach.restype = hsa_status_t
    hsa_amd_ipc_memory_detach.argtypes = [ctypes.POINTER(None)]
except AttributeError:
    pass
hsa_amd_ipc_signal_t = struct_hsa_amd_ipc_memory_s
try:
    hsa_amd_ipc_signal_create = _libraries['libhsa-runtime64.so'].hsa_amd_ipc_signal_create
    hsa_amd_ipc_signal_create.restype = hsa_status_t
    hsa_amd_ipc_signal_create.argtypes = [hsa_signal_t, ctypes.POINTER(struct_hsa_amd_ipc_memory_s)]
except AttributeError:
    pass
try:
    hsa_amd_ipc_signal_attach = _libraries['libhsa-runtime64.so'].hsa_amd_ipc_signal_attach
    hsa_amd_ipc_signal_attach.restype = hsa_status_t
    hsa_amd_ipc_signal_attach.argtypes = [ctypes.POINTER(struct_hsa_amd_ipc_memory_s), ctypes.POINTER(struct_hsa_signal_s)]
except AttributeError:
    pass

# values for enumeration 'hsa_amd_event_type_s'
hsa_amd_event_type_s__enumvalues = {
    0: 'HSA_AMD_GPU_MEMORY_FAULT_EVENT',
    1: 'HSA_AMD_GPU_HW_EXCEPTION_EVENT',
}
HSA_AMD_GPU_MEMORY_FAULT_EVENT = 0
HSA_AMD_GPU_HW_EXCEPTION_EVENT = 1
hsa_amd_event_type_s = ctypes.c_uint32 # enum
hsa_amd_event_type_t = hsa_amd_event_type_s
hsa_amd_event_type_t__enumvalues = hsa_amd_event_type_s__enumvalues

# values for enumeration 'hsa_amd_memory_fault_reason_t'
hsa_amd_memory_fault_reason_t__enumvalues = {
    1: 'HSA_AMD_MEMORY_FAULT_PAGE_NOT_PRESENT',
    2: 'HSA_AMD_MEMORY_FAULT_READ_ONLY',
    4: 'HSA_AMD_MEMORY_FAULT_NX',
    8: 'HSA_AMD_MEMORY_FAULT_HOST_ONLY',
    16: 'HSA_AMD_MEMORY_FAULT_DRAMECC',
    32: 'HSA_AMD_MEMORY_FAULT_IMPRECISE',
    64: 'HSA_AMD_MEMORY_FAULT_SRAMECC',
    -2147483648: 'HSA_AMD_MEMORY_FAULT_HANG',
}
HSA_AMD_MEMORY_FAULT_PAGE_NOT_PRESENT = 1
HSA_AMD_MEMORY_FAULT_READ_ONLY = 2
HSA_AMD_MEMORY_FAULT_NX = 4
HSA_AMD_MEMORY_FAULT_HOST_ONLY = 8
HSA_AMD_MEMORY_FAULT_DRAMECC = 16
HSA_AMD_MEMORY_FAULT_IMPRECISE = 32
HSA_AMD_MEMORY_FAULT_SRAMECC = 64
HSA_AMD_MEMORY_FAULT_HANG = -2147483648
hsa_amd_memory_fault_reason_t = ctypes.c_int32 # enum
class struct_hsa_amd_gpu_memory_fault_info_s(Structure):
    pass

struct_hsa_amd_gpu_memory_fault_info_s._pack_ = 1 # source:False
struct_hsa_amd_gpu_memory_fault_info_s._fields_ = [
    ('agent', hsa_agent_t),
    ('virtual_address', ctypes.c_uint64),
    ('fault_reason_mask', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

hsa_amd_gpu_memory_fault_info_t = struct_hsa_amd_gpu_memory_fault_info_s

# values for enumeration 'hsa_amd_hw_exception_reset_type_t'
hsa_amd_hw_exception_reset_type_t__enumvalues = {
    1: 'HSA_AMD_HW_EXCEPTION_RESET_TYPE_OTHER',
}
HSA_AMD_HW_EXCEPTION_RESET_TYPE_OTHER = 1
hsa_amd_hw_exception_reset_type_t = ctypes.c_uint32 # enum

# values for enumeration 'hsa_amd_hw_exception_reset_cause_t'
hsa_amd_hw_exception_reset_cause_t__enumvalues = {
    1: 'HSA_AMD_HW_EXCEPTION_CAUSE_GPU_HANG',
    2: 'HSA_AMD_HW_EXCEPTION_CAUSE_ECC',
}
HSA_AMD_HW_EXCEPTION_CAUSE_GPU_HANG = 1
HSA_AMD_HW_EXCEPTION_CAUSE_ECC = 2
hsa_amd_hw_exception_reset_cause_t = ctypes.c_uint32 # enum
class struct_hsa_amd_gpu_hw_exception_info_s(Structure):
    _pack_ = 1 # source:False
    _fields_ = [
    ('agent', hsa_agent_t),
    ('reset_type', hsa_amd_hw_exception_reset_type_t),
    ('reset_cause', hsa_amd_hw_exception_reset_cause_t),
     ]

hsa_amd_gpu_hw_exception_info_t = struct_hsa_amd_gpu_hw_exception_info_s
class struct_hsa_amd_event_s(Structure):
    pass

class union_union_hsa_ext_amd_h_2329(Union):
    pass

union_union_hsa_ext_amd_h_2329._pack_ = 1 # source:False
union_union_hsa_ext_amd_h_2329._fields_ = [
    ('memory_fault', hsa_amd_gpu_memory_fault_info_t),
    ('hw_exception', hsa_amd_gpu_hw_exception_info_t),
    ('PADDING_0', ctypes.c_ubyte * 8),
]

struct_hsa_amd_event_s._pack_ = 1 # source:False
struct_hsa_amd_event_s._anonymous_ = ('_0',)
struct_hsa_amd_event_s._fields_ = [
    ('event_type', hsa_amd_event_type_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('_0', union_union_hsa_ext_amd_h_2329),
]

hsa_amd_event_t = struct_hsa_amd_event_s
hsa_amd_system_event_callback_t = ctypes.CFUNCTYPE(hsa_status_t, ctypes.POINTER(struct_hsa_amd_event_s), ctypes.POINTER(None))
try:
    hsa_amd_register_system_event_handler = _libraries['libhsa-runtime64.so'].hsa_amd_register_system_event_handler
    hsa_amd_register_system_event_handler.restype = hsa_status_t
    hsa_amd_register_system_event_handler.argtypes = [hsa_amd_system_event_callback_t, ctypes.POINTER(None)]
except AttributeError:
    pass

# values for enumeration 'hsa_amd_queue_priority_s'
hsa_amd_queue_priority_s__enumvalues = {
    0: 'HSA_AMD_QUEUE_PRIORITY_LOW',
    1: 'HSA_AMD_QUEUE_PRIORITY_NORMAL',
    2: 'HSA_AMD_QUEUE_PRIORITY_HIGH',
}
HSA_AMD_QUEUE_PRIORITY_LOW = 0
HSA_AMD_QUEUE_PRIORITY_NORMAL = 1
HSA_AMD_QUEUE_PRIORITY_HIGH = 2
hsa_amd_queue_priority_s = ctypes.c_uint32 # enum
hsa_amd_queue_priority_t = hsa_amd_queue_priority_s
hsa_amd_queue_priority_t__enumvalues = hsa_amd_queue_priority_s__enumvalues
try:
    hsa_amd_queue_set_priority = _libraries['libhsa-runtime64.so'].hsa_amd_queue_set_priority
    hsa_amd_queue_set_priority.restype = hsa_status_t
    hsa_amd_queue_set_priority.argtypes = [ctypes.POINTER(struct_hsa_queue_s), hsa_amd_queue_priority_t]
except AttributeError:
    pass
hsa_amd_deallocation_callback_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(None), ctypes.POINTER(None))
try:
    hsa_amd_register_deallocation_callback = _libraries['libhsa-runtime64.so'].hsa_amd_register_deallocation_callback
    hsa_amd_register_deallocation_callback.restype = hsa_status_t
    hsa_amd_register_deallocation_callback.argtypes = [ctypes.POINTER(None), hsa_amd_deallocation_callback_t, ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_amd_deregister_deallocation_callback = _libraries['libhsa-runtime64.so'].hsa_amd_deregister_deallocation_callback
    hsa_amd_deregister_deallocation_callback.restype = hsa_status_t
    hsa_amd_deregister_deallocation_callback.argtypes = [ctypes.POINTER(None), hsa_amd_deallocation_callback_t]
except AttributeError:
    pass

# values for enumeration 'hsa_amd_svm_model_s'
hsa_amd_svm_model_s__enumvalues = {
    0: 'HSA_AMD_SVM_GLOBAL_FLAG_FINE_GRAINED',
    1: 'HSA_AMD_SVM_GLOBAL_FLAG_COARSE_GRAINED',
    2: 'HSA_AMD_SVM_GLOBAL_FLAG_INDETERMINATE',
}
HSA_AMD_SVM_GLOBAL_FLAG_FINE_GRAINED = 0
HSA_AMD_SVM_GLOBAL_FLAG_COARSE_GRAINED = 1
HSA_AMD_SVM_GLOBAL_FLAG_INDETERMINATE = 2
hsa_amd_svm_model_s = ctypes.c_uint32 # enum
hsa_amd_svm_model_t = hsa_amd_svm_model_s
hsa_amd_svm_model_t__enumvalues = hsa_amd_svm_model_s__enumvalues

# values for enumeration 'hsa_amd_svm_attribute_s'
hsa_amd_svm_attribute_s__enumvalues = {
    0: 'HSA_AMD_SVM_ATTRIB_GLOBAL_FLAG',
    1: 'HSA_AMD_SVM_ATTRIB_READ_ONLY',
    2: 'HSA_AMD_SVM_ATTRIB_HIVE_LOCAL',
    3: 'HSA_AMD_SVM_ATTRIB_MIGRATION_GRANULARITY',
    4: 'HSA_AMD_SVM_ATTRIB_PREFERRED_LOCATION',
    5: 'HSA_AMD_SVM_ATTRIB_PREFETCH_LOCATION',
    6: 'HSA_AMD_SVM_ATTRIB_READ_MOSTLY',
    7: 'HSA_AMD_SVM_ATTRIB_GPU_EXEC',
    512: 'HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE',
    513: 'HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE_IN_PLACE',
    514: 'HSA_AMD_SVM_ATTRIB_AGENT_NO_ACCESS',
    515: 'HSA_AMD_SVM_ATTRIB_ACCESS_QUERY',
}
HSA_AMD_SVM_ATTRIB_GLOBAL_FLAG = 0
HSA_AMD_SVM_ATTRIB_READ_ONLY = 1
HSA_AMD_SVM_ATTRIB_HIVE_LOCAL = 2
HSA_AMD_SVM_ATTRIB_MIGRATION_GRANULARITY = 3
HSA_AMD_SVM_ATTRIB_PREFERRED_LOCATION = 4
HSA_AMD_SVM_ATTRIB_PREFETCH_LOCATION = 5
HSA_AMD_SVM_ATTRIB_READ_MOSTLY = 6
HSA_AMD_SVM_ATTRIB_GPU_EXEC = 7
HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE = 512
HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE_IN_PLACE = 513
HSA_AMD_SVM_ATTRIB_AGENT_NO_ACCESS = 514
HSA_AMD_SVM_ATTRIB_ACCESS_QUERY = 515
hsa_amd_svm_attribute_s = ctypes.c_uint32 # enum
hsa_amd_svm_attribute_t = hsa_amd_svm_attribute_s
hsa_amd_svm_attribute_t__enumvalues = hsa_amd_svm_attribute_s__enumvalues
class struct_hsa_amd_svm_attribute_pair_s(Structure):
    pass

struct_hsa_amd_svm_attribute_pair_s._pack_ = 1 # source:False
struct_hsa_amd_svm_attribute_pair_s._fields_ = [
    ('attribute', ctypes.c_uint64),
    ('value', ctypes.c_uint64),
]

hsa_amd_svm_attribute_pair_t = struct_hsa_amd_svm_attribute_pair_s
try:
    hsa_amd_svm_attributes_set = _libraries['libhsa-runtime64.so'].hsa_amd_svm_attributes_set
    hsa_amd_svm_attributes_set.restype = hsa_status_t
    hsa_amd_svm_attributes_set.argtypes = [ctypes.POINTER(None), size_t, ctypes.POINTER(struct_hsa_amd_svm_attribute_pair_s), size_t]
except AttributeError:
    pass
try:
    hsa_amd_svm_attributes_get = _libraries['libhsa-runtime64.so'].hsa_amd_svm_attributes_get
    hsa_amd_svm_attributes_get.restype = hsa_status_t
    hsa_amd_svm_attributes_get.argtypes = [ctypes.POINTER(None), size_t, ctypes.POINTER(struct_hsa_amd_svm_attribute_pair_s), size_t]
except AttributeError:
    pass
try:
    hsa_amd_svm_prefetch_async = _libraries['libhsa-runtime64.so'].hsa_amd_svm_prefetch_async
    hsa_amd_svm_prefetch_async.restype = hsa_status_t
    hsa_amd_svm_prefetch_async.argtypes = [ctypes.POINTER(None), size_t, hsa_agent_t, uint32_t, ctypes.POINTER(struct_hsa_signal_s), hsa_signal_t]
except AttributeError:
    pass
try:
    hsa_amd_spm_acquire = _libraries['libhsa-runtime64.so'].hsa_amd_spm_acquire
    hsa_amd_spm_acquire.restype = hsa_status_t
    hsa_amd_spm_acquire.argtypes = [hsa_agent_t]
except AttributeError:
    pass
try:
    hsa_amd_spm_release = _libraries['libhsa-runtime64.so'].hsa_amd_spm_release
    hsa_amd_spm_release.restype = hsa_status_t
    hsa_amd_spm_release.argtypes = [hsa_agent_t]
except AttributeError:
    pass
try:
    hsa_amd_spm_set_dest_buffer = _libraries['libhsa-runtime64.so'].hsa_amd_spm_set_dest_buffer
    hsa_amd_spm_set_dest_buffer.restype = hsa_status_t
    hsa_amd_spm_set_dest_buffer.argtypes = [hsa_agent_t, size_t, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(None), ctypes.POINTER(ctypes.c_bool)]
except AttributeError:
    pass
try:
    hsa_amd_portable_export_dmabuf = _libraries['libhsa-runtime64.so'].hsa_amd_portable_export_dmabuf
    hsa_amd_portable_export_dmabuf.restype = hsa_status_t
    hsa_amd_portable_export_dmabuf.argtypes = [ctypes.POINTER(None), size_t, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_uint64)]
except AttributeError:
    pass
try:
    hsa_amd_portable_close_dmabuf = _libraries['libhsa-runtime64.so'].hsa_amd_portable_close_dmabuf
    hsa_amd_portable_close_dmabuf.restype = hsa_status_t
    hsa_amd_portable_close_dmabuf.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    hsa_amd_vmem_address_reserve = _libraries['libhsa-runtime64.so'].hsa_amd_vmem_address_reserve
    hsa_amd_vmem_address_reserve.restype = hsa_status_t
    hsa_amd_vmem_address_reserve.argtypes = [ctypes.POINTER(ctypes.POINTER(None)), size_t, uint64_t, uint64_t]
except AttributeError:
    pass
try:
    hsa_amd_vmem_address_free = _libraries['libhsa-runtime64.so'].hsa_amd_vmem_address_free
    hsa_amd_vmem_address_free.restype = hsa_status_t
    hsa_amd_vmem_address_free.argtypes = [ctypes.POINTER(None), size_t]
except AttributeError:
    pass
class struct_hsa_amd_vmem_alloc_handle_s(Structure):
    pass

struct_hsa_amd_vmem_alloc_handle_s._pack_ = 1 # source:False
struct_hsa_amd_vmem_alloc_handle_s._fields_ = [
    ('handle', ctypes.c_uint64),
]

hsa_amd_vmem_alloc_handle_t = struct_hsa_amd_vmem_alloc_handle_s

# values for enumeration 'hsa_amd_memory_type_t'
hsa_amd_memory_type_t__enumvalues = {
    0: 'MEMORY_TYPE_NONE',
    1: 'MEMORY_TYPE_PINNED',
}
MEMORY_TYPE_NONE = 0
MEMORY_TYPE_PINNED = 1
hsa_amd_memory_type_t = ctypes.c_uint32 # enum
try:
    hsa_amd_vmem_handle_create = _libraries['libhsa-runtime64.so'].hsa_amd_vmem_handle_create
    hsa_amd_vmem_handle_create.restype = hsa_status_t
    hsa_amd_vmem_handle_create.argtypes = [hsa_amd_memory_pool_t, size_t, hsa_amd_memory_type_t, uint64_t, ctypes.POINTER(struct_hsa_amd_vmem_alloc_handle_s)]
except AttributeError:
    pass
try:
    hsa_amd_vmem_handle_release = _libraries['libhsa-runtime64.so'].hsa_amd_vmem_handle_release
    hsa_amd_vmem_handle_release.restype = hsa_status_t
    hsa_amd_vmem_handle_release.argtypes = [hsa_amd_vmem_alloc_handle_t]
except AttributeError:
    pass
try:
    hsa_amd_vmem_map = _libraries['libhsa-runtime64.so'].hsa_amd_vmem_map
    hsa_amd_vmem_map.restype = hsa_status_t
    hsa_amd_vmem_map.argtypes = [ctypes.POINTER(None), size_t, size_t, hsa_amd_vmem_alloc_handle_t, uint64_t]
except AttributeError:
    pass
try:
    hsa_amd_vmem_unmap = _libraries['libhsa-runtime64.so'].hsa_amd_vmem_unmap
    hsa_amd_vmem_unmap.restype = hsa_status_t
    hsa_amd_vmem_unmap.argtypes = [ctypes.POINTER(None), size_t]
except AttributeError:
    pass
class struct_hsa_amd_memory_access_desc_s(Structure):
    pass

struct_hsa_amd_memory_access_desc_s._pack_ = 1 # source:False
struct_hsa_amd_memory_access_desc_s._fields_ = [
    ('permissions', hsa_access_permission_t),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('agent_handle', hsa_agent_t),
]

hsa_amd_memory_access_desc_t = struct_hsa_amd_memory_access_desc_s
try:
    hsa_amd_vmem_set_access = _libraries['libhsa-runtime64.so'].hsa_amd_vmem_set_access
    hsa_amd_vmem_set_access.restype = hsa_status_t
    hsa_amd_vmem_set_access.argtypes = [ctypes.POINTER(None), size_t, ctypes.POINTER(struct_hsa_amd_memory_access_desc_s), size_t]
except AttributeError:
    pass
try:
    hsa_amd_vmem_get_access = _libraries['libhsa-runtime64.so'].hsa_amd_vmem_get_access
    hsa_amd_vmem_get_access.restype = hsa_status_t
    hsa_amd_vmem_get_access.argtypes = [ctypes.POINTER(None), ctypes.POINTER(hsa_access_permission_t), hsa_agent_t]
except AttributeError:
    pass
try:
    hsa_amd_vmem_export_shareable_handle = _libraries['libhsa-runtime64.so'].hsa_amd_vmem_export_shareable_handle
    hsa_amd_vmem_export_shareable_handle.restype = hsa_status_t
    hsa_amd_vmem_export_shareable_handle.argtypes = [ctypes.POINTER(ctypes.c_int32), hsa_amd_vmem_alloc_handle_t, uint64_t]
except AttributeError:
    pass
try:
    hsa_amd_vmem_import_shareable_handle = _libraries['libhsa-runtime64.so'].hsa_amd_vmem_import_shareable_handle
    hsa_amd_vmem_import_shareable_handle.restype = hsa_status_t
    hsa_amd_vmem_import_shareable_handle.argtypes = [ctypes.c_int32, ctypes.POINTER(struct_hsa_amd_vmem_alloc_handle_s)]
except AttributeError:
    pass
try:
    hsa_amd_vmem_retain_alloc_handle = _libraries['libhsa-runtime64.so'].hsa_amd_vmem_retain_alloc_handle
    hsa_amd_vmem_retain_alloc_handle.restype = hsa_status_t
    hsa_amd_vmem_retain_alloc_handle.argtypes = [ctypes.POINTER(struct_hsa_amd_vmem_alloc_handle_s), ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    hsa_amd_vmem_get_alloc_properties_from_handle = _libraries['libhsa-runtime64.so'].hsa_amd_vmem_get_alloc_properties_from_handle
    hsa_amd_vmem_get_alloc_properties_from_handle.restype = hsa_status_t
    hsa_amd_vmem_get_alloc_properties_from_handle.argtypes = [hsa_amd_vmem_alloc_handle_t, ctypes.POINTER(struct_hsa_amd_memory_pool_s), ctypes.POINTER(hsa_amd_memory_type_t)]
except AttributeError:
    pass
class struct_BrigModuleHeader(Structure):
    pass

BrigModule_t = ctypes.POINTER(struct_BrigModuleHeader)

# values for enumeration 'enum_hsa_ext_finalize_h_69'
enum_hsa_ext_finalize_h_69__enumvalues = {
    8192: 'HSA_EXT_STATUS_ERROR_INVALID_PROGRAM',
    8193: 'HSA_EXT_STATUS_ERROR_INVALID_MODULE',
    8194: 'HSA_EXT_STATUS_ERROR_INCOMPATIBLE_MODULE',
    8195: 'HSA_EXT_STATUS_ERROR_MODULE_ALREADY_INCLUDED',
    8196: 'HSA_EXT_STATUS_ERROR_SYMBOL_MISMATCH',
    8197: 'HSA_EXT_STATUS_ERROR_FINALIZATION_FAILED',
    8198: 'HSA_EXT_STATUS_ERROR_DIRECTIVE_MISMATCH',
}
HSA_EXT_STATUS_ERROR_INVALID_PROGRAM = 8192
HSA_EXT_STATUS_ERROR_INVALID_MODULE = 8193
HSA_EXT_STATUS_ERROR_INCOMPATIBLE_MODULE = 8194
HSA_EXT_STATUS_ERROR_MODULE_ALREADY_INCLUDED = 8195
HSA_EXT_STATUS_ERROR_SYMBOL_MISMATCH = 8196
HSA_EXT_STATUS_ERROR_FINALIZATION_FAILED = 8197
HSA_EXT_STATUS_ERROR_DIRECTIVE_MISMATCH = 8198
enum_hsa_ext_finalize_h_69 = ctypes.c_uint32 # enum
hsa_ext_module_t = ctypes.POINTER(struct_BrigModuleHeader)
class struct_hsa_ext_program_s(Structure):
    pass

struct_hsa_ext_program_s._pack_ = 1 # source:False
struct_hsa_ext_program_s._fields_ = [
    ('handle', ctypes.c_uint64),
]

hsa_ext_program_t = struct_hsa_ext_program_s
try:
    hsa_ext_program_create = _libraries['libhsa-runtime64.so'].hsa_ext_program_create
    hsa_ext_program_create.restype = hsa_status_t
    hsa_ext_program_create.argtypes = [hsa_machine_model_t, hsa_profile_t, hsa_default_float_rounding_mode_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct_hsa_ext_program_s)]
except AttributeError:
    pass
try:
    hsa_ext_program_destroy = _libraries['libhsa-runtime64.so'].hsa_ext_program_destroy
    hsa_ext_program_destroy.restype = hsa_status_t
    hsa_ext_program_destroy.argtypes = [hsa_ext_program_t]
except AttributeError:
    pass
try:
    hsa_ext_program_add_module = _libraries['libhsa-runtime64.so'].hsa_ext_program_add_module
    hsa_ext_program_add_module.restype = hsa_status_t
    hsa_ext_program_add_module.argtypes = [hsa_ext_program_t, hsa_ext_module_t]
except AttributeError:
    pass
try:
    hsa_ext_program_iterate_modules = _libraries['libhsa-runtime64.so'].hsa_ext_program_iterate_modules
    hsa_ext_program_iterate_modules.restype = hsa_status_t
    hsa_ext_program_iterate_modules.argtypes = [hsa_ext_program_t, ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_ext_program_s, ctypes.POINTER(struct_BrigModuleHeader), ctypes.POINTER(None)), ctypes.POINTER(None)]
except AttributeError:
    pass

# values for enumeration 'hsa_ext_program_info_t'
hsa_ext_program_info_t__enumvalues = {
    0: 'HSA_EXT_PROGRAM_INFO_MACHINE_MODEL',
    1: 'HSA_EXT_PROGRAM_INFO_PROFILE',
    2: 'HSA_EXT_PROGRAM_INFO_DEFAULT_FLOAT_ROUNDING_MODE',
}
HSA_EXT_PROGRAM_INFO_MACHINE_MODEL = 0
HSA_EXT_PROGRAM_INFO_PROFILE = 1
HSA_EXT_PROGRAM_INFO_DEFAULT_FLOAT_ROUNDING_MODE = 2
hsa_ext_program_info_t = ctypes.c_uint32 # enum
try:
    hsa_ext_program_get_info = _libraries['libhsa-runtime64.so'].hsa_ext_program_get_info
    hsa_ext_program_get_info.restype = hsa_status_t
    hsa_ext_program_get_info.argtypes = [hsa_ext_program_t, hsa_ext_program_info_t, ctypes.POINTER(None)]
except AttributeError:
    pass

# values for enumeration 'hsa_ext_finalizer_call_convention_t'
hsa_ext_finalizer_call_convention_t__enumvalues = {
    -1: 'HSA_EXT_FINALIZER_CALL_CONVENTION_AUTO',
}
HSA_EXT_FINALIZER_CALL_CONVENTION_AUTO = -1
hsa_ext_finalizer_call_convention_t = ctypes.c_int32 # enum
class struct_hsa_ext_control_directives_s(Structure):
    pass

struct_hsa_ext_control_directives_s._pack_ = 1 # source:False
struct_hsa_ext_control_directives_s._fields_ = [
    ('control_directives_mask', ctypes.c_uint64),
    ('break_exceptions_mask', ctypes.c_uint16),
    ('detect_exceptions_mask', ctypes.c_uint16),
    ('max_dynamic_group_size', ctypes.c_uint32),
    ('max_flat_grid_size', ctypes.c_uint64),
    ('max_flat_workgroup_size', ctypes.c_uint32),
    ('reserved1', ctypes.c_uint32),
    ('required_grid_size', ctypes.c_uint64 * 3),
    ('required_workgroup_size', hsa_dim3_t),
    ('required_dim', ctypes.c_ubyte),
    ('reserved2', ctypes.c_ubyte * 75),
]

hsa_ext_control_directives_t = struct_hsa_ext_control_directives_s
try:
    hsa_ext_program_finalize = _libraries['libhsa-runtime64.so'].hsa_ext_program_finalize
    hsa_ext_program_finalize.restype = hsa_status_t
    hsa_ext_program_finalize.argtypes = [hsa_ext_program_t, hsa_isa_t, int32_t, hsa_ext_control_directives_t, ctypes.POINTER(ctypes.c_char), hsa_code_object_type_t, ctypes.POINTER(struct_hsa_code_object_s)]
except AttributeError:
    pass
class struct_hsa_ext_finalizer_1_00_pfn_s(Structure):
    pass

struct_hsa_ext_finalizer_1_00_pfn_s._pack_ = 1 # source:False
struct_hsa_ext_finalizer_1_00_pfn_s._fields_ = [
    ('hsa_ext_program_create', ctypes.CFUNCTYPE(hsa_status_t, hsa_machine_model_t, hsa_profile_t, hsa_default_float_rounding_mode_t, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct_hsa_ext_program_s))),
    ('hsa_ext_program_destroy', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_ext_program_s)),
    ('hsa_ext_program_add_module', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_ext_program_s, ctypes.POINTER(struct_BrigModuleHeader))),
    ('hsa_ext_program_iterate_modules', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_ext_program_s, ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_ext_program_s, ctypes.POINTER(struct_BrigModuleHeader), ctypes.POINTER(None)), ctypes.POINTER(None))),
    ('hsa_ext_program_get_info', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_ext_program_s, hsa_ext_program_info_t, ctypes.POINTER(None))),
    ('hsa_ext_program_finalize', ctypes.CFUNCTYPE(hsa_status_t, struct_hsa_ext_program_s, struct_hsa_isa_s, ctypes.c_int32, struct_hsa_ext_control_directives_s, ctypes.POINTER(ctypes.c_char), hsa_code_object_type_t, ctypes.POINTER(struct_hsa_code_object_s))),
]

hsa_ext_finalizer_1_00_pfn_t = struct_hsa_ext_finalizer_1_00_pfn_s
__all__ = \
    ['BrigModule_t', 'HSA_ACCESS_PERMISSION_NONE',
    'HSA_ACCESS_PERMISSION_RO', 'HSA_ACCESS_PERMISSION_RW',
    'HSA_ACCESS_PERMISSION_WO', 'HSA_AGENT_FEATURE_AGENT_DISPATCH',
    'HSA_AGENT_FEATURE_KERNEL_DISPATCH',
    'HSA_AGENT_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES',
    'HSA_AGENT_INFO_CACHE_SIZE',
    'HSA_AGENT_INFO_DEFAULT_FLOAT_ROUNDING_MODE',
    'HSA_AGENT_INFO_DEVICE', 'HSA_AGENT_INFO_EXTENSIONS',
    'HSA_AGENT_INFO_FAST_F16_OPERATION',
    'HSA_AGENT_INFO_FBARRIER_MAX_SIZE', 'HSA_AGENT_INFO_FEATURE',
    'HSA_AGENT_INFO_GRID_MAX_DIM', 'HSA_AGENT_INFO_GRID_MAX_SIZE',
    'HSA_AGENT_INFO_ISA', 'HSA_AGENT_INFO_LAST',
    'HSA_AGENT_INFO_MACHINE_MODEL', 'HSA_AGENT_INFO_NAME',
    'HSA_AGENT_INFO_NODE', 'HSA_AGENT_INFO_PROFILE',
    'HSA_AGENT_INFO_QUEUES_MAX', 'HSA_AGENT_INFO_QUEUE_MAX_SIZE',
    'HSA_AGENT_INFO_QUEUE_MIN_SIZE', 'HSA_AGENT_INFO_QUEUE_TYPE',
    'HSA_AGENT_INFO_VENDOR_NAME', 'HSA_AGENT_INFO_VERSION_MAJOR',
    'HSA_AGENT_INFO_VERSION_MINOR', 'HSA_AGENT_INFO_WAVEFRONT_SIZE',
    'HSA_AGENT_INFO_WORKGROUP_MAX_DIM',
    'HSA_AGENT_INFO_WORKGROUP_MAX_SIZE',
    'HSA_AMD_AGENT_INFO_ASIC_FAMILY_ID',
    'HSA_AMD_AGENT_INFO_ASIC_REVISION', 'HSA_AMD_AGENT_INFO_BDFID',
    'HSA_AMD_AGENT_INFO_CACHELINE_SIZE', 'HSA_AMD_AGENT_INFO_CHIP_ID',
    'HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT',
    'HSA_AMD_AGENT_INFO_COOPERATIVE_COMPUTE_UNIT_COUNT',
    'HSA_AMD_AGENT_INFO_COOPERATIVE_QUEUES',
    'HSA_AMD_AGENT_INFO_DOMAIN', 'HSA_AMD_AGENT_INFO_DRIVER_NODE_ID',
    'HSA_AMD_AGENT_INFO_DRIVER_UID', 'HSA_AMD_AGENT_INFO_HDP_FLUSH',
    'HSA_AMD_AGENT_INFO_IOMMU_SUPPORT',
    'HSA_AMD_AGENT_INFO_MAX_ADDRESS_WATCH_POINTS',
    'HSA_AMD_AGENT_INFO_MAX_CLOCK_FREQUENCY',
    'HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU',
    'HSA_AMD_AGENT_INFO_MEMORY_AVAIL',
    'HSA_AMD_AGENT_INFO_MEMORY_MAX_FREQUENCY',
    'HSA_AMD_AGENT_INFO_MEMORY_WIDTH',
    'HSA_AMD_AGENT_INFO_NEAREST_CPU',
    'HSA_AMD_AGENT_INFO_NUM_SDMA_ENG',
    'HSA_AMD_AGENT_INFO_NUM_SDMA_XGMI_ENG',
    'HSA_AMD_AGENT_INFO_NUM_SHADER_ARRAYS_PER_SE',
    'HSA_AMD_AGENT_INFO_NUM_SHADER_ENGINES',
    'HSA_AMD_AGENT_INFO_NUM_SIMDS_PER_CU',
    'HSA_AMD_AGENT_INFO_NUM_XCC', 'HSA_AMD_AGENT_INFO_PRODUCT_NAME',
    'HSA_AMD_AGENT_INFO_SDMA_UCODE_VERSION',
    'HSA_AMD_AGENT_INFO_SVM_DIRECT_HOST_ACCESS',
    'HSA_AMD_AGENT_INFO_TIMESTAMP_FREQUENCY',
    'HSA_AMD_AGENT_INFO_UCODE_VERSION', 'HSA_AMD_AGENT_INFO_UUID',
    'HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS',
    'HSA_AMD_AGENT_MEMORY_POOL_INFO_LINK_INFO',
    'HSA_AMD_AGENT_MEMORY_POOL_INFO_NUM_LINK_HOPS',
    'HSA_AMD_COHERENCY_TYPE_COHERENT',
    'HSA_AMD_COHERENCY_TYPE_NONCOHERENT', 'HSA_AMD_FIRST_EXTENSION',
    'HSA_AMD_GPU_HW_EXCEPTION_EVENT',
    'HSA_AMD_GPU_MEMORY_FAULT_EVENT',
    'HSA_AMD_HW_EXCEPTION_CAUSE_ECC',
    'HSA_AMD_HW_EXCEPTION_CAUSE_GPU_HANG',
    'HSA_AMD_HW_EXCEPTION_RESET_TYPE_OTHER', 'HSA_AMD_LAST_EXTENSION',
    'HSA_AMD_LINK_INFO_TYPE_HYPERTRANSPORT',
    'HSA_AMD_LINK_INFO_TYPE_INFINBAND', 'HSA_AMD_LINK_INFO_TYPE_PCIE',
    'HSA_AMD_LINK_INFO_TYPE_QPI', 'HSA_AMD_LINK_INFO_TYPE_XGMI',
    'HSA_AMD_MEMORY_FAULT_DRAMECC', 'HSA_AMD_MEMORY_FAULT_HANG',
    'HSA_AMD_MEMORY_FAULT_HOST_ONLY',
    'HSA_AMD_MEMORY_FAULT_IMPRECISE', 'HSA_AMD_MEMORY_FAULT_NX',
    'HSA_AMD_MEMORY_FAULT_PAGE_NOT_PRESENT',
    'HSA_AMD_MEMORY_FAULT_READ_ONLY', 'HSA_AMD_MEMORY_FAULT_SRAMECC',
    'HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT',
    'HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT',
    'HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED',
    'HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED',
    'HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED',
    'HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED',
    'HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT',
    'HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL',
    'HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE',
    'HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS',
    'HSA_AMD_MEMORY_POOL_INFO_LOCATION',
    'HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALIGNMENT',
    'HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED',
    'HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE',
    'HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE',
    'HSA_AMD_MEMORY_POOL_INFO_SEGMENT',
    'HSA_AMD_MEMORY_POOL_INFO_SIZE',
    'HSA_AMD_MEMORY_POOL_LOCATION_CPU',
    'HSA_AMD_MEMORY_POOL_LOCATION_GPU',
    'HSA_AMD_MEMORY_POOL_PCIE_FLAG',
    'HSA_AMD_MEMORY_POOL_STANDARD_FLAG',
    'HSA_AMD_PACKET_TYPE_BARRIER_VALUE',
    'HSA_AMD_QUEUE_PRIORITY_HIGH', 'HSA_AMD_QUEUE_PRIORITY_LOW',
    'HSA_AMD_QUEUE_PRIORITY_NORMAL', 'HSA_AMD_REGION_INFO_BASE',
    'HSA_AMD_REGION_INFO_BUS_WIDTH',
    'HSA_AMD_REGION_INFO_HOST_ACCESSIBLE',
    'HSA_AMD_REGION_INFO_MAX_CLOCK_FREQUENCY',
    'HSA_AMD_SDMA_ENGINE_0', 'HSA_AMD_SDMA_ENGINE_1',
    'HSA_AMD_SDMA_ENGINE_10', 'HSA_AMD_SDMA_ENGINE_11',
    'HSA_AMD_SDMA_ENGINE_12', 'HSA_AMD_SDMA_ENGINE_13',
    'HSA_AMD_SDMA_ENGINE_14', 'HSA_AMD_SDMA_ENGINE_15',
    'HSA_AMD_SDMA_ENGINE_2', 'HSA_AMD_SDMA_ENGINE_3',
    'HSA_AMD_SDMA_ENGINE_4', 'HSA_AMD_SDMA_ENGINE_5',
    'HSA_AMD_SDMA_ENGINE_6', 'HSA_AMD_SDMA_ENGINE_7',
    'HSA_AMD_SDMA_ENGINE_8', 'HSA_AMD_SDMA_ENGINE_9',
    'HSA_AMD_SEGMENT_GLOBAL', 'HSA_AMD_SEGMENT_GROUP',
    'HSA_AMD_SEGMENT_PRIVATE', 'HSA_AMD_SEGMENT_READONLY',
    'HSA_AMD_SIGNAL_AMD_GPU_ONLY', 'HSA_AMD_SIGNAL_IPC',
    'HSA_AMD_SVM_ATTRIB_ACCESS_QUERY',
    'HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE',
    'HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE_IN_PLACE',
    'HSA_AMD_SVM_ATTRIB_AGENT_NO_ACCESS',
    'HSA_AMD_SVM_ATTRIB_GLOBAL_FLAG', 'HSA_AMD_SVM_ATTRIB_GPU_EXEC',
    'HSA_AMD_SVM_ATTRIB_HIVE_LOCAL',
    'HSA_AMD_SVM_ATTRIB_MIGRATION_GRANULARITY',
    'HSA_AMD_SVM_ATTRIB_PREFERRED_LOCATION',
    'HSA_AMD_SVM_ATTRIB_PREFETCH_LOCATION',
    'HSA_AMD_SVM_ATTRIB_READ_MOSTLY', 'HSA_AMD_SVM_ATTRIB_READ_ONLY',
    'HSA_AMD_SVM_GLOBAL_FLAG_COARSE_GRAINED',
    'HSA_AMD_SVM_GLOBAL_FLAG_FINE_GRAINED',
    'HSA_AMD_SVM_GLOBAL_FLAG_INDETERMINATE',
    'HSA_AMD_SYSTEM_INFO_BUILD_VERSION',
    'HSA_AMD_SYSTEM_INFO_DMABUF_SUPPORTED',
    'HSA_AMD_SYSTEM_INFO_MWAITX_ENABLED',
    'HSA_AMD_SYSTEM_INFO_SVM_ACCESSIBLE_BY_DEFAULT',
    'HSA_AMD_SYSTEM_INFO_SVM_SUPPORTED',
    'HSA_AMD_SYSTEM_INFO_VIRTUAL_MEM_API_SUPPORTED',
    'HSA_AMD_SYSTEM_INFO_XNACK_ENABLED', 'HSA_CACHE_INFO_LEVEL',
    'HSA_CACHE_INFO_NAME', 'HSA_CACHE_INFO_NAME_LENGTH',
    'HSA_CACHE_INFO_SIZE',
    'HSA_CODE_OBJECT_INFO_DEFAULT_FLOAT_ROUNDING_MODE',
    'HSA_CODE_OBJECT_INFO_ISA', 'HSA_CODE_OBJECT_INFO_MACHINE_MODEL',
    'HSA_CODE_OBJECT_INFO_PROFILE', 'HSA_CODE_OBJECT_INFO_TYPE',
    'HSA_CODE_OBJECT_INFO_VERSION', 'HSA_CODE_OBJECT_TYPE_PROGRAM',
    'HSA_CODE_SYMBOL_INFO_INDIRECT_FUNCTION_CALL_CONVENTION',
    'HSA_CODE_SYMBOL_INFO_IS_DEFINITION',
    'HSA_CODE_SYMBOL_INFO_KERNEL_CALL_CONVENTION',
    'HSA_CODE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK',
    'HSA_CODE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE',
    'HSA_CODE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT',
    'HSA_CODE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE',
    'HSA_CODE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE',
    'HSA_CODE_SYMBOL_INFO_LINKAGE',
    'HSA_CODE_SYMBOL_INFO_MODULE_NAME',
    'HSA_CODE_SYMBOL_INFO_MODULE_NAME_LENGTH',
    'HSA_CODE_SYMBOL_INFO_NAME', 'HSA_CODE_SYMBOL_INFO_NAME_LENGTH',
    'HSA_CODE_SYMBOL_INFO_TYPE',
    'HSA_CODE_SYMBOL_INFO_VARIABLE_ALIGNMENT',
    'HSA_CODE_SYMBOL_INFO_VARIABLE_ALLOCATION',
    'HSA_CODE_SYMBOL_INFO_VARIABLE_IS_CONST',
    'HSA_CODE_SYMBOL_INFO_VARIABLE_SEGMENT',
    'HSA_CODE_SYMBOL_INFO_VARIABLE_SIZE',
    'HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT',
    'HSA_DEFAULT_FLOAT_ROUNDING_MODE_NEAR',
    'HSA_DEFAULT_FLOAT_ROUNDING_MODE_ZERO', 'HSA_DEVICE_TYPE_CPU',
    'HSA_DEVICE_TYPE_DSP', 'HSA_DEVICE_TYPE_GPU',
    'HSA_ENDIANNESS_BIG', 'HSA_ENDIANNESS_LITTLE',
    'HSA_EXCEPTION_POLICY_BREAK', 'HSA_EXCEPTION_POLICY_DETECT',
    'HSA_EXECUTABLE_INFO_DEFAULT_FLOAT_ROUNDING_MODE',
    'HSA_EXECUTABLE_INFO_PROFILE', 'HSA_EXECUTABLE_INFO_STATE',
    'HSA_EXECUTABLE_STATE_FROZEN', 'HSA_EXECUTABLE_STATE_UNFROZEN',
    'HSA_EXECUTABLE_SYMBOL_INFO_AGENT',
    'HSA_EXECUTABLE_SYMBOL_INFO_INDIRECT_FUNCTION_CALL_CONVENTION',
    'HSA_EXECUTABLE_SYMBOL_INFO_INDIRECT_FUNCTION_OBJECT',
    'HSA_EXECUTABLE_SYMBOL_INFO_IS_DEFINITION',
    'HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_CALL_CONVENTION',
    'HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK',
    'HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE',
    'HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT',
    'HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE',
    'HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT',
    'HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE',
    'HSA_EXECUTABLE_SYMBOL_INFO_LINKAGE',
    'HSA_EXECUTABLE_SYMBOL_INFO_MODULE_NAME',
    'HSA_EXECUTABLE_SYMBOL_INFO_MODULE_NAME_LENGTH',
    'HSA_EXECUTABLE_SYMBOL_INFO_NAME',
    'HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH',
    'HSA_EXECUTABLE_SYMBOL_INFO_TYPE',
    'HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS',
    'HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ALIGNMENT',
    'HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ALLOCATION',
    'HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_IS_CONST',
    'HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SEGMENT',
    'HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SIZE',
    'HSA_EXTENSION_AMD_AQLPROFILE', 'HSA_EXTENSION_AMD_LOADER',
    'HSA_EXTENSION_AMD_PROFILER', 'HSA_EXTENSION_FINALIZER',
    'HSA_EXTENSION_IMAGES', 'HSA_EXTENSION_PERFORMANCE_COUNTERS',
    'HSA_EXTENSION_PROFILING_EVENTS', 'HSA_EXTENSION_STD_LAST',
    'HSA_EXT_AGENT_INFO_IMAGE_1DA_MAX_ELEMENTS',
    'HSA_EXT_AGENT_INFO_IMAGE_1DB_MAX_ELEMENTS',
    'HSA_EXT_AGENT_INFO_IMAGE_1D_MAX_ELEMENTS',
    'HSA_EXT_AGENT_INFO_IMAGE_2DADEPTH_MAX_ELEMENTS',
    'HSA_EXT_AGENT_INFO_IMAGE_2DA_MAX_ELEMENTS',
    'HSA_EXT_AGENT_INFO_IMAGE_2DDEPTH_MAX_ELEMENTS',
    'HSA_EXT_AGENT_INFO_IMAGE_2D_MAX_ELEMENTS',
    'HSA_EXT_AGENT_INFO_IMAGE_3D_MAX_ELEMENTS',
    'HSA_EXT_AGENT_INFO_IMAGE_ARRAY_MAX_LAYERS',
    'HSA_EXT_AGENT_INFO_IMAGE_LINEAR_ROW_PITCH_ALIGNMENT',
    'HSA_EXT_AGENT_INFO_MAX_IMAGE_RD_HANDLES',
    'HSA_EXT_AGENT_INFO_MAX_IMAGE_RORW_HANDLES',
    'HSA_EXT_AGENT_INFO_MAX_SAMPLER_HANDLERS',
    'HSA_EXT_FINALIZER_CALL_CONVENTION_AUTO',
    'HSA_EXT_IMAGE_CAPABILITY_ACCESS_INVARIANT_DATA_LAYOUT',
    'HSA_EXT_IMAGE_CAPABILITY_NOT_SUPPORTED',
    'HSA_EXT_IMAGE_CAPABILITY_READ_MODIFY_WRITE',
    'HSA_EXT_IMAGE_CAPABILITY_READ_ONLY',
    'HSA_EXT_IMAGE_CAPABILITY_READ_WRITE',
    'HSA_EXT_IMAGE_CAPABILITY_WRITE_ONLY',
    'HSA_EXT_IMAGE_CHANNEL_ORDER_A',
    'HSA_EXT_IMAGE_CHANNEL_ORDER_ABGR',
    'HSA_EXT_IMAGE_CHANNEL_ORDER_ARGB',
    'HSA_EXT_IMAGE_CHANNEL_ORDER_BGRA',
    'HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH',
    'HSA_EXT_IMAGE_CHANNEL_ORDER_DEPTH_STENCIL',
    'HSA_EXT_IMAGE_CHANNEL_ORDER_INTENSITY',
    'HSA_EXT_IMAGE_CHANNEL_ORDER_LUMINANCE',
    'HSA_EXT_IMAGE_CHANNEL_ORDER_R', 'HSA_EXT_IMAGE_CHANNEL_ORDER_RA',
    'HSA_EXT_IMAGE_CHANNEL_ORDER_RG',
    'HSA_EXT_IMAGE_CHANNEL_ORDER_RGB',
    'HSA_EXT_IMAGE_CHANNEL_ORDER_RGBA',
    'HSA_EXT_IMAGE_CHANNEL_ORDER_RGBX',
    'HSA_EXT_IMAGE_CHANNEL_ORDER_RGX',
    'HSA_EXT_IMAGE_CHANNEL_ORDER_RX',
    'HSA_EXT_IMAGE_CHANNEL_ORDER_SBGRA',
    'HSA_EXT_IMAGE_CHANNEL_ORDER_SRGB',
    'HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBA',
    'HSA_EXT_IMAGE_CHANNEL_ORDER_SRGBX',
    'HSA_EXT_IMAGE_CHANNEL_TYPE_FLOAT',
    'HSA_EXT_IMAGE_CHANNEL_TYPE_HALF_FLOAT',
    'HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT16',
    'HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT32',
    'HSA_EXT_IMAGE_CHANNEL_TYPE_SIGNED_INT8',
    'HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT16',
    'HSA_EXT_IMAGE_CHANNEL_TYPE_SNORM_INT8',
    'HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT16',
    'HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT24',
    'HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_INT8',
    'HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_101010',
    'HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_555',
    'HSA_EXT_IMAGE_CHANNEL_TYPE_UNORM_SHORT_565',
    'HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT16',
    'HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT32',
    'HSA_EXT_IMAGE_CHANNEL_TYPE_UNSIGNED_INT8',
    'HSA_EXT_IMAGE_DATA_LAYOUT_LINEAR',
    'HSA_EXT_IMAGE_DATA_LAYOUT_OPAQUE', 'HSA_EXT_IMAGE_GEOMETRY_1D',
    'HSA_EXT_IMAGE_GEOMETRY_1DA', 'HSA_EXT_IMAGE_GEOMETRY_1DB',
    'HSA_EXT_IMAGE_GEOMETRY_2D', 'HSA_EXT_IMAGE_GEOMETRY_2DA',
    'HSA_EXT_IMAGE_GEOMETRY_2DADEPTH',
    'HSA_EXT_IMAGE_GEOMETRY_2DDEPTH', 'HSA_EXT_IMAGE_GEOMETRY_3D',
    'HSA_EXT_POINTER_TYPE_GRAPHICS', 'HSA_EXT_POINTER_TYPE_HSA',
    'HSA_EXT_POINTER_TYPE_IPC', 'HSA_EXT_POINTER_TYPE_LOCKED',
    'HSA_EXT_POINTER_TYPE_UNKNOWN',
    'HSA_EXT_PROGRAM_INFO_DEFAULT_FLOAT_ROUNDING_MODE',
    'HSA_EXT_PROGRAM_INFO_MACHINE_MODEL',
    'HSA_EXT_PROGRAM_INFO_PROFILE',
    'HSA_EXT_SAMPLER_ADDRESSING_MODE_CLAMP_TO_BORDER',
    'HSA_EXT_SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE',
    'HSA_EXT_SAMPLER_ADDRESSING_MODE_MIRRORED_REPEAT',
    'HSA_EXT_SAMPLER_ADDRESSING_MODE_REPEAT',
    'HSA_EXT_SAMPLER_ADDRESSING_MODE_UNDEFINED',
    'HSA_EXT_SAMPLER_COORDINATE_MODE_NORMALIZED',
    'HSA_EXT_SAMPLER_COORDINATE_MODE_UNNORMALIZED',
    'HSA_EXT_SAMPLER_FILTER_MODE_LINEAR',
    'HSA_EXT_SAMPLER_FILTER_MODE_NEAREST',
    'HSA_EXT_STATUS_ERROR_DIRECTIVE_MISMATCH',
    'HSA_EXT_STATUS_ERROR_FINALIZATION_FAILED',
    'HSA_EXT_STATUS_ERROR_IMAGE_FORMAT_UNSUPPORTED',
    'HSA_EXT_STATUS_ERROR_IMAGE_PITCH_UNSUPPORTED',
    'HSA_EXT_STATUS_ERROR_IMAGE_SIZE_UNSUPPORTED',
    'HSA_EXT_STATUS_ERROR_INCOMPATIBLE_MODULE',
    'HSA_EXT_STATUS_ERROR_INVALID_MODULE',
    'HSA_EXT_STATUS_ERROR_INVALID_PROGRAM',
    'HSA_EXT_STATUS_ERROR_MODULE_ALREADY_INCLUDED',
    'HSA_EXT_STATUS_ERROR_SAMPLER_DESCRIPTOR_UNSUPPORTED',
    'HSA_EXT_STATUS_ERROR_SYMBOL_MISMATCH', 'HSA_FENCE_SCOPE_AGENT',
    'HSA_FENCE_SCOPE_NONE', 'HSA_FENCE_SCOPE_SYSTEM',
    'HSA_FLUSH_MODE_FTZ', 'HSA_FLUSH_MODE_NON_FTZ', 'HSA_FP_TYPE_16',
    'HSA_FP_TYPE_32', 'HSA_FP_TYPE_64', 'HSA_IOMMU_SUPPORT_NONE',
    'HSA_IOMMU_SUPPORT_V2',
    'HSA_ISA_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES',
    'HSA_ISA_INFO_CALL_CONVENTION_COUNT',
    'HSA_ISA_INFO_CALL_CONVENTION_INFO_WAVEFRONTS_PER_COMPUTE_UNIT',
    'HSA_ISA_INFO_CALL_CONVENTION_INFO_WAVEFRONT_SIZE',
    'HSA_ISA_INFO_DEFAULT_FLOAT_ROUNDING_MODES',
    'HSA_ISA_INFO_FAST_F16_OPERATION',
    'HSA_ISA_INFO_FBARRIER_MAX_SIZE', 'HSA_ISA_INFO_GRID_MAX_DIM',
    'HSA_ISA_INFO_GRID_MAX_SIZE', 'HSA_ISA_INFO_MACHINE_MODELS',
    'HSA_ISA_INFO_NAME', 'HSA_ISA_INFO_NAME_LENGTH',
    'HSA_ISA_INFO_PROFILES', 'HSA_ISA_INFO_WORKGROUP_MAX_DIM',
    'HSA_ISA_INFO_WORKGROUP_MAX_SIZE',
    'HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS',
    'HSA_KERNEL_DISPATCH_PACKET_SETUP_WIDTH_DIMENSIONS',
    'HSA_MACHINE_MODEL_LARGE', 'HSA_MACHINE_MODEL_SMALL',
    'HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE',
    'HSA_PACKET_HEADER_BARRIER',
    'HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE',
    'HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE',
    'HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE',
    'HSA_PACKET_HEADER_TYPE',
    'HSA_PACKET_HEADER_WIDTH_ACQUIRE_FENCE_SCOPE',
    'HSA_PACKET_HEADER_WIDTH_BARRIER',
    'HSA_PACKET_HEADER_WIDTH_RELEASE_FENCE_SCOPE',
    'HSA_PACKET_HEADER_WIDTH_SCACQUIRE_FENCE_SCOPE',
    'HSA_PACKET_HEADER_WIDTH_SCRELEASE_FENCE_SCOPE',
    'HSA_PACKET_HEADER_WIDTH_TYPE', 'HSA_PACKET_TYPE_AGENT_DISPATCH',
    'HSA_PACKET_TYPE_BARRIER_AND', 'HSA_PACKET_TYPE_BARRIER_OR',
    'HSA_PACKET_TYPE_INVALID', 'HSA_PACKET_TYPE_KERNEL_DISPATCH',
    'HSA_PACKET_TYPE_VENDOR_SPECIFIC', 'HSA_PROFILE_BASE',
    'HSA_PROFILE_FULL', 'HSA_QUEUE_FEATURE_AGENT_DISPATCH',
    'HSA_QUEUE_FEATURE_KERNEL_DISPATCH', 'HSA_QUEUE_TYPE_COOPERATIVE',
    'HSA_QUEUE_TYPE_MULTI', 'HSA_QUEUE_TYPE_SINGLE',
    'HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED',
    'HSA_REGION_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED',
    'HSA_REGION_GLOBAL_FLAG_FINE_GRAINED',
    'HSA_REGION_GLOBAL_FLAG_KERNARG',
    'HSA_REGION_INFO_ALLOC_MAX_PRIVATE_WORKGROUP_SIZE',
    'HSA_REGION_INFO_ALLOC_MAX_SIZE', 'HSA_REGION_INFO_GLOBAL_FLAGS',
    'HSA_REGION_INFO_RUNTIME_ALLOC_ALIGNMENT',
    'HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED',
    'HSA_REGION_INFO_RUNTIME_ALLOC_GRANULE',
    'HSA_REGION_INFO_SEGMENT', 'HSA_REGION_INFO_SIZE',
    'HSA_REGION_SEGMENT_GLOBAL', 'HSA_REGION_SEGMENT_GROUP',
    'HSA_REGION_SEGMENT_KERNARG', 'HSA_REGION_SEGMENT_PRIVATE',
    'HSA_REGION_SEGMENT_READONLY', 'HSA_ROUND_METHOD_DOUBLE',
    'HSA_ROUND_METHOD_SINGLE', 'HSA_SIGNAL_CONDITION_EQ',
    'HSA_SIGNAL_CONDITION_GTE', 'HSA_SIGNAL_CONDITION_LT',
    'HSA_SIGNAL_CONDITION_NE', 'HSA_STATUS_CU_MASK_REDUCED',
    'HSA_STATUS_ERROR', 'HSA_STATUS_ERROR_EXCEPTION',
    'HSA_STATUS_ERROR_FATAL', 'HSA_STATUS_ERROR_FROZEN_EXECUTABLE',
    'HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION',
    'HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS',
    'HSA_STATUS_ERROR_INVALID_AGENT',
    'HSA_STATUS_ERROR_INVALID_ALLOCATION',
    'HSA_STATUS_ERROR_INVALID_ARGUMENT',
    'HSA_STATUS_ERROR_INVALID_CACHE',
    'HSA_STATUS_ERROR_INVALID_CODE_OBJECT',
    'HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER',
    'HSA_STATUS_ERROR_INVALID_CODE_SYMBOL',
    'HSA_STATUS_ERROR_INVALID_EXECUTABLE',
    'HSA_STATUS_ERROR_INVALID_EXECUTABLE_SYMBOL',
    'HSA_STATUS_ERROR_INVALID_FILE', 'HSA_STATUS_ERROR_INVALID_INDEX',
    'HSA_STATUS_ERROR_INVALID_ISA',
    'HSA_STATUS_ERROR_INVALID_ISA_NAME',
    'HSA_STATUS_ERROR_INVALID_MEMORY_POOL',
    'HSA_STATUS_ERROR_INVALID_PACKET_FORMAT',
    'HSA_STATUS_ERROR_INVALID_QUEUE',
    'HSA_STATUS_ERROR_INVALID_QUEUE_CREATION',
    'HSA_STATUS_ERROR_INVALID_REGION',
    'HSA_STATUS_ERROR_INVALID_RUNTIME_STATE',
    'HSA_STATUS_ERROR_INVALID_SIGNAL',
    'HSA_STATUS_ERROR_INVALID_SIGNAL_GROUP',
    'HSA_STATUS_ERROR_INVALID_SYMBOL_NAME',
    'HSA_STATUS_ERROR_INVALID_WAVEFRONT',
    'HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION',
    'HSA_STATUS_ERROR_MEMORY_FAULT',
    'HSA_STATUS_ERROR_NOT_INITIALIZED',
    'HSA_STATUS_ERROR_OUT_OF_REGISTERS',
    'HSA_STATUS_ERROR_OUT_OF_RESOURCES',
    'HSA_STATUS_ERROR_REFCOUNT_OVERFLOW',
    'HSA_STATUS_ERROR_RESOURCE_FREE',
    'HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED',
    'HSA_STATUS_ERROR_VARIABLE_UNDEFINED', 'HSA_STATUS_INFO_BREAK',
    'HSA_STATUS_SUCCESS', 'HSA_SYMBOL_KIND_INDIRECT_FUNCTION',
    'HSA_SYMBOL_KIND_KERNEL', 'HSA_SYMBOL_KIND_VARIABLE',
    'HSA_SYMBOL_LINKAGE_MODULE', 'HSA_SYMBOL_LINKAGE_PROGRAM',
    'HSA_SYSTEM_INFO_ENDIANNESS', 'HSA_SYSTEM_INFO_EXTENSIONS',
    'HSA_SYSTEM_INFO_MACHINE_MODEL',
    'HSA_SYSTEM_INFO_SIGNAL_MAX_WAIT', 'HSA_SYSTEM_INFO_TIMESTAMP',
    'HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY',
    'HSA_SYSTEM_INFO_VERSION_MAJOR', 'HSA_SYSTEM_INFO_VERSION_MINOR',
    'HSA_VARIABLE_ALLOCATION_AGENT',
    'HSA_VARIABLE_ALLOCATION_PROGRAM', 'HSA_VARIABLE_SEGMENT_GLOBAL',
    'HSA_VARIABLE_SEGMENT_READONLY', 'HSA_WAIT_STATE_ACTIVE',
    'HSA_WAIT_STATE_BLOCKED', 'HSA_WAVEFRONT_INFO_SIZE',
    'MEMORY_TYPE_NONE', 'MEMORY_TYPE_PINNED',
    'enum_hsa_ext_amd_h_179', 'enum_hsa_ext_finalize_h_69',
    'enum_hsa_ext_image_h_68', 'enum_hsa_ext_image_h_93',
    'hsaDeviceToDevice', 'hsaDeviceToHost', 'hsaHostToDevice',
    'hsaHostToHost', 'hsa_access_permission_t',
    'hsa_agent_dispatch_packet_t', 'hsa_agent_extension_supported',
    'hsa_agent_feature_t', 'hsa_agent_get_exception_policies',
    'hsa_agent_get_info', 'hsa_agent_info_t',
    'hsa_agent_iterate_caches', 'hsa_agent_iterate_isas',
    'hsa_agent_iterate_regions',
    'hsa_agent_major_extension_supported', 'hsa_agent_t',
    'hsa_amd_agent_info_s', 'hsa_amd_agent_info_t',
    'hsa_amd_agent_info_t__enumvalues',
    'hsa_amd_agent_iterate_memory_pools',
    'hsa_amd_agent_memory_pool_get_info',
    'hsa_amd_agent_memory_pool_info_t', 'hsa_amd_agents_allow_access',
    'hsa_amd_async_function', 'hsa_amd_barrier_value_packet_t',
    'hsa_amd_coherency_get_type', 'hsa_amd_coherency_set_type',
    'hsa_amd_coherency_type_s', 'hsa_amd_coherency_type_t',
    'hsa_amd_coherency_type_t__enumvalues',
    'hsa_amd_copy_direction_t', 'hsa_amd_deallocation_callback_t',
    'hsa_amd_deregister_deallocation_callback', 'hsa_amd_event_t',
    'hsa_amd_event_type_s', 'hsa_amd_event_type_t',
    'hsa_amd_event_type_t__enumvalues',
    'hsa_amd_gpu_hw_exception_info_t',
    'hsa_amd_gpu_memory_fault_info_t', 'hsa_amd_hdp_flush_t',
    'hsa_amd_hw_exception_reset_cause_t',
    'hsa_amd_hw_exception_reset_type_t', 'hsa_amd_image_create',
    'hsa_amd_image_descriptor_t', 'hsa_amd_image_get_info_max_dim',
    'hsa_amd_interop_map_buffer', 'hsa_amd_interop_unmap_buffer',
    'hsa_amd_iommu_version_t', 'hsa_amd_ipc_memory_attach',
    'hsa_amd_ipc_memory_create', 'hsa_amd_ipc_memory_detach',
    'hsa_amd_ipc_memory_t', 'hsa_amd_ipc_signal_attach',
    'hsa_amd_ipc_signal_create', 'hsa_amd_ipc_signal_t',
    'hsa_amd_link_info_type_t', 'hsa_amd_memory_access_desc_t',
    'hsa_amd_memory_async_copy',
    'hsa_amd_memory_async_copy_on_engine',
    'hsa_amd_memory_async_copy_rect',
    'hsa_amd_memory_copy_engine_status',
    'hsa_amd_memory_fault_reason_t', 'hsa_amd_memory_fill',
    'hsa_amd_memory_lock', 'hsa_amd_memory_lock_to_pool',
    'hsa_amd_memory_migrate', 'hsa_amd_memory_pool_access_t',
    'hsa_amd_memory_pool_allocate', 'hsa_amd_memory_pool_can_migrate',
    'hsa_amd_memory_pool_flag_s', 'hsa_amd_memory_pool_flag_t',
    'hsa_amd_memory_pool_flag_t__enumvalues',
    'hsa_amd_memory_pool_free', 'hsa_amd_memory_pool_get_info',
    'hsa_amd_memory_pool_global_flag_s',
    'hsa_amd_memory_pool_global_flag_t',
    'hsa_amd_memory_pool_global_flag_t__enumvalues',
    'hsa_amd_memory_pool_info_t', 'hsa_amd_memory_pool_link_info_t',
    'hsa_amd_memory_pool_location_s',
    'hsa_amd_memory_pool_location_t',
    'hsa_amd_memory_pool_location_t__enumvalues',
    'hsa_amd_memory_pool_t', 'hsa_amd_memory_type_t',
    'hsa_amd_memory_unlock', 'hsa_amd_packet_type8_t',
    'hsa_amd_packet_type_t', 'hsa_amd_pointer_info',
    'hsa_amd_pointer_info_set_userdata', 'hsa_amd_pointer_info_t',
    'hsa_amd_pointer_type_t', 'hsa_amd_portable_close_dmabuf',
    'hsa_amd_portable_export_dmabuf',
    'hsa_amd_profiling_async_copy_enable',
    'hsa_amd_profiling_async_copy_time_t',
    'hsa_amd_profiling_convert_tick_to_system_domain',
    'hsa_amd_profiling_dispatch_time_t',
    'hsa_amd_profiling_get_async_copy_time',
    'hsa_amd_profiling_get_dispatch_time',
    'hsa_amd_profiling_set_profiler_enabled',
    'hsa_amd_queue_cu_get_mask', 'hsa_amd_queue_cu_set_mask',
    'hsa_amd_queue_priority_s', 'hsa_amd_queue_priority_t',
    'hsa_amd_queue_priority_t__enumvalues',
    'hsa_amd_queue_set_priority', 'hsa_amd_region_info_s',
    'hsa_amd_region_info_t', 'hsa_amd_region_info_t__enumvalues',
    'hsa_amd_register_deallocation_callback',
    'hsa_amd_register_system_event_handler', 'hsa_amd_sdma_engine_id',
    'hsa_amd_sdma_engine_id_t',
    'hsa_amd_sdma_engine_id_t__enumvalues', 'hsa_amd_segment_t',
    'hsa_amd_signal_async_handler', 'hsa_amd_signal_attribute_t',
    'hsa_amd_signal_create', 'hsa_amd_signal_handler',
    'hsa_amd_signal_value_pointer', 'hsa_amd_signal_wait_any',
    'hsa_amd_spm_acquire', 'hsa_amd_spm_release',
    'hsa_amd_spm_set_dest_buffer', 'hsa_amd_svm_attribute_pair_t',
    'hsa_amd_svm_attribute_s', 'hsa_amd_svm_attribute_t',
    'hsa_amd_svm_attribute_t__enumvalues',
    'hsa_amd_svm_attributes_get', 'hsa_amd_svm_attributes_set',
    'hsa_amd_svm_model_s', 'hsa_amd_svm_model_t',
    'hsa_amd_svm_model_t__enumvalues', 'hsa_amd_svm_prefetch_async',
    'hsa_amd_system_event_callback_t',
    'hsa_amd_vendor_packet_header_t', 'hsa_amd_vmem_address_free',
    'hsa_amd_vmem_address_reserve', 'hsa_amd_vmem_alloc_handle_t',
    'hsa_amd_vmem_export_shareable_handle', 'hsa_amd_vmem_get_access',
    'hsa_amd_vmem_get_alloc_properties_from_handle',
    'hsa_amd_vmem_handle_create', 'hsa_amd_vmem_handle_release',
    'hsa_amd_vmem_import_shareable_handle', 'hsa_amd_vmem_map',
    'hsa_amd_vmem_retain_alloc_handle', 'hsa_amd_vmem_set_access',
    'hsa_amd_vmem_unmap', 'hsa_barrier_and_packet_t',
    'hsa_barrier_or_packet_t', 'hsa_cache_get_info',
    'hsa_cache_info_t', 'hsa_cache_t', 'hsa_callback_data_t',
    'hsa_code_object_deserialize', 'hsa_code_object_destroy',
    'hsa_code_object_get_info', 'hsa_code_object_get_symbol',
    'hsa_code_object_get_symbol_from_name', 'hsa_code_object_info_t',
    'hsa_code_object_iterate_symbols',
    'hsa_code_object_reader_create_from_file',
    'hsa_code_object_reader_create_from_memory',
    'hsa_code_object_reader_destroy', 'hsa_code_object_reader_t',
    'hsa_code_object_serialize', 'hsa_code_object_t',
    'hsa_code_object_type_t', 'hsa_code_symbol_get_info',
    'hsa_code_symbol_info_t', 'hsa_code_symbol_t',
    'hsa_default_float_rounding_mode_t', 'hsa_device_type_t',
    'hsa_dim3_t', 'hsa_endianness_t', 'hsa_exception_policy_t',
    'hsa_executable_agent_global_variable_define',
    'hsa_executable_create', 'hsa_executable_create_alt',
    'hsa_executable_destroy', 'hsa_executable_freeze',
    'hsa_executable_get_info', 'hsa_executable_get_symbol',
    'hsa_executable_get_symbol_by_name',
    'hsa_executable_global_variable_define', 'hsa_executable_info_t',
    'hsa_executable_iterate_agent_symbols',
    'hsa_executable_iterate_program_symbols',
    'hsa_executable_iterate_symbols',
    'hsa_executable_load_agent_code_object',
    'hsa_executable_load_code_object',
    'hsa_executable_load_program_code_object',
    'hsa_executable_readonly_variable_define',
    'hsa_executable_state_t', 'hsa_executable_symbol_get_info',
    'hsa_executable_symbol_info_t', 'hsa_executable_symbol_t',
    'hsa_executable_t', 'hsa_executable_validate',
    'hsa_executable_validate_alt', 'hsa_ext_control_directives_t',
    'hsa_ext_finalizer_1_00_pfn_t',
    'hsa_ext_finalizer_call_convention_t',
    'hsa_ext_image_capability_t', 'hsa_ext_image_channel_order32_t',
    'hsa_ext_image_channel_order_t', 'hsa_ext_image_channel_type32_t',
    'hsa_ext_image_channel_type_t', 'hsa_ext_image_clear',
    'hsa_ext_image_copy', 'hsa_ext_image_create',
    'hsa_ext_image_create_with_layout', 'hsa_ext_image_data_get_info',
    'hsa_ext_image_data_get_info_with_layout',
    'hsa_ext_image_data_info_t', 'hsa_ext_image_data_layout_t',
    'hsa_ext_image_descriptor_t', 'hsa_ext_image_destroy',
    'hsa_ext_image_export', 'hsa_ext_image_format_t',
    'hsa_ext_image_geometry_t', 'hsa_ext_image_get_capability',
    'hsa_ext_image_get_capability_with_layout',
    'hsa_ext_image_import', 'hsa_ext_image_region_t',
    'hsa_ext_image_t', 'hsa_ext_images_1_00_pfn_t',
    'hsa_ext_images_1_pfn_t', 'hsa_ext_module_t',
    'hsa_ext_program_add_module', 'hsa_ext_program_create',
    'hsa_ext_program_destroy', 'hsa_ext_program_finalize',
    'hsa_ext_program_get_info', 'hsa_ext_program_info_t',
    'hsa_ext_program_iterate_modules', 'hsa_ext_program_t',
    'hsa_ext_sampler_addressing_mode32_t',
    'hsa_ext_sampler_addressing_mode_t',
    'hsa_ext_sampler_coordinate_mode32_t',
    'hsa_ext_sampler_coordinate_mode_t', 'hsa_ext_sampler_create',
    'hsa_ext_sampler_descriptor_t', 'hsa_ext_sampler_destroy',
    'hsa_ext_sampler_filter_mode32_t',
    'hsa_ext_sampler_filter_mode_t', 'hsa_ext_sampler_t',
    'hsa_extension_get_name', 'hsa_extension_t', 'hsa_fence_scope_t',
    'hsa_file_t', 'hsa_flush_mode_t', 'hsa_fp_type_t', 'hsa_init',
    'hsa_isa_compatible', 'hsa_isa_from_name',
    'hsa_isa_get_exception_policies', 'hsa_isa_get_info',
    'hsa_isa_get_info_alt', 'hsa_isa_get_round_method',
    'hsa_isa_info_t', 'hsa_isa_iterate_wavefronts', 'hsa_isa_t',
    'hsa_iterate_agents', 'hsa_kernel_dispatch_packet_setup_t',
    'hsa_kernel_dispatch_packet_setup_width_t',
    'hsa_kernel_dispatch_packet_t', 'hsa_loaded_code_object_t',
    'hsa_machine_model_t', 'hsa_memory_allocate',
    'hsa_memory_assign_agent', 'hsa_memory_copy',
    'hsa_memory_deregister', 'hsa_memory_free', 'hsa_memory_register',
    'hsa_packet_header_t', 'hsa_packet_header_width_t',
    'hsa_packet_type_t', 'hsa_pitched_ptr_t', 'hsa_profile_t',
    'hsa_queue_add_write_index_acq_rel',
    'hsa_queue_add_write_index_acquire',
    'hsa_queue_add_write_index_relaxed',
    'hsa_queue_add_write_index_release',
    'hsa_queue_add_write_index_scacq_screl',
    'hsa_queue_add_write_index_scacquire',
    'hsa_queue_add_write_index_screlease',
    'hsa_queue_cas_write_index_acq_rel',
    'hsa_queue_cas_write_index_acquire',
    'hsa_queue_cas_write_index_relaxed',
    'hsa_queue_cas_write_index_release',
    'hsa_queue_cas_write_index_scacq_screl',
    'hsa_queue_cas_write_index_scacquire',
    'hsa_queue_cas_write_index_screlease', 'hsa_queue_create',
    'hsa_queue_destroy', 'hsa_queue_feature_t',
    'hsa_queue_inactivate', 'hsa_queue_load_read_index_acquire',
    'hsa_queue_load_read_index_relaxed',
    'hsa_queue_load_read_index_scacquire',
    'hsa_queue_load_write_index_acquire',
    'hsa_queue_load_write_index_relaxed',
    'hsa_queue_load_write_index_scacquire',
    'hsa_queue_store_read_index_relaxed',
    'hsa_queue_store_read_index_release',
    'hsa_queue_store_read_index_screlease',
    'hsa_queue_store_write_index_relaxed',
    'hsa_queue_store_write_index_release',
    'hsa_queue_store_write_index_screlease', 'hsa_queue_t',
    'hsa_queue_type32_t', 'hsa_queue_type_t', 'hsa_region_get_info',
    'hsa_region_global_flag_t', 'hsa_region_info_t',
    'hsa_region_segment_t', 'hsa_region_t', 'hsa_round_method_t',
    'hsa_shut_down', 'hsa_signal_add_acq_rel',
    'hsa_signal_add_acquire', 'hsa_signal_add_relaxed',
    'hsa_signal_add_release', 'hsa_signal_add_scacq_screl',
    'hsa_signal_add_scacquire', 'hsa_signal_add_screlease',
    'hsa_signal_and_acq_rel', 'hsa_signal_and_acquire',
    'hsa_signal_and_relaxed', 'hsa_signal_and_release',
    'hsa_signal_and_scacq_screl', 'hsa_signal_and_scacquire',
    'hsa_signal_and_screlease', 'hsa_signal_cas_acq_rel',
    'hsa_signal_cas_acquire', 'hsa_signal_cas_relaxed',
    'hsa_signal_cas_release', 'hsa_signal_cas_scacq_screl',
    'hsa_signal_cas_scacquire', 'hsa_signal_cas_screlease',
    'hsa_signal_condition32_t', 'hsa_signal_condition_t',
    'hsa_signal_create', 'hsa_signal_destroy',
    'hsa_signal_exchange_acq_rel', 'hsa_signal_exchange_acquire',
    'hsa_signal_exchange_relaxed', 'hsa_signal_exchange_release',
    'hsa_signal_exchange_scacq_screl',
    'hsa_signal_exchange_scacquire', 'hsa_signal_exchange_screlease',
    'hsa_signal_group_create', 'hsa_signal_group_destroy',
    'hsa_signal_group_t', 'hsa_signal_group_wait_any_relaxed',
    'hsa_signal_group_wait_any_scacquire', 'hsa_signal_load_acquire',
    'hsa_signal_load_relaxed', 'hsa_signal_load_scacquire',
    'hsa_signal_or_acq_rel', 'hsa_signal_or_acquire',
    'hsa_signal_or_relaxed', 'hsa_signal_or_release',
    'hsa_signal_or_scacq_screl', 'hsa_signal_or_scacquire',
    'hsa_signal_or_screlease', 'hsa_signal_silent_store_relaxed',
    'hsa_signal_silent_store_screlease', 'hsa_signal_store_relaxed',
    'hsa_signal_store_release', 'hsa_signal_store_screlease',
    'hsa_signal_subtract_acq_rel', 'hsa_signal_subtract_acquire',
    'hsa_signal_subtract_relaxed', 'hsa_signal_subtract_release',
    'hsa_signal_subtract_scacq_screl',
    'hsa_signal_subtract_scacquire', 'hsa_signal_subtract_screlease',
    'hsa_signal_t', 'hsa_signal_value_t', 'hsa_signal_wait_acquire',
    'hsa_signal_wait_relaxed', 'hsa_signal_wait_scacquire',
    'hsa_signal_xor_acq_rel', 'hsa_signal_xor_acquire',
    'hsa_signal_xor_relaxed', 'hsa_signal_xor_release',
    'hsa_signal_xor_scacq_screl', 'hsa_signal_xor_scacquire',
    'hsa_signal_xor_screlease', 'hsa_soft_queue_create',
    'hsa_status_string', 'hsa_status_t', 'hsa_symbol_kind_t',
    'hsa_symbol_linkage_t', 'hsa_system_extension_supported',
    'hsa_system_get_extension_table', 'hsa_system_get_info',
    'hsa_system_get_major_extension_table', 'hsa_system_info_t',
    'hsa_system_major_extension_supported',
    'hsa_variable_allocation_t', 'hsa_variable_segment_t',
    'hsa_wait_state_t', 'hsa_wavefront_get_info',
    'hsa_wavefront_info_t', 'hsa_wavefront_t', 'int32_t', 'size_t',
    'struct_BrigModuleHeader', 'struct_hsa_agent_dispatch_packet_s',
    'struct_hsa_agent_s', 'struct_hsa_amd_barrier_value_packet_s',
    'struct_hsa_amd_event_s',
    'struct_hsa_amd_gpu_hw_exception_info_s',
    'struct_hsa_amd_gpu_memory_fault_info_s',
    'struct_hsa_amd_hdp_flush_s', 'struct_hsa_amd_image_descriptor_s',
    'struct_hsa_amd_ipc_memory_s',
    'struct_hsa_amd_memory_access_desc_s',
    'struct_hsa_amd_memory_pool_link_info_s',
    'struct_hsa_amd_memory_pool_s', 'struct_hsa_amd_packet_header_s',
    'struct_hsa_amd_pointer_info_s',
    'struct_hsa_amd_profiling_async_copy_time_s',
    'struct_hsa_amd_profiling_dispatch_time_s',
    'struct_hsa_amd_svm_attribute_pair_s',
    'struct_hsa_amd_vmem_alloc_handle_s',
    'struct_hsa_barrier_and_packet_s',
    'struct_hsa_barrier_or_packet_s', 'struct_hsa_cache_s',
    'struct_hsa_callback_data_s', 'struct_hsa_code_object_reader_s',
    'struct_hsa_code_object_s', 'struct_hsa_code_symbol_s',
    'struct_hsa_dim3_s', 'struct_hsa_executable_s',
    'struct_hsa_executable_symbol_s',
    'struct_hsa_ext_control_directives_s',
    'struct_hsa_ext_finalizer_1_00_pfn_s',
    'struct_hsa_ext_image_data_info_s',
    'struct_hsa_ext_image_descriptor_s',
    'struct_hsa_ext_image_format_s', 'struct_hsa_ext_image_region_s',
    'struct_hsa_ext_image_s', 'struct_hsa_ext_images_1_00_pfn_s',
    'struct_hsa_ext_images_1_pfn_s', 'struct_hsa_ext_program_s',
    'struct_hsa_ext_sampler_descriptor_s', 'struct_hsa_ext_sampler_s',
    'struct_hsa_isa_s', 'struct_hsa_kernel_dispatch_packet_s',
    'struct_hsa_loaded_code_object_s', 'struct_hsa_pitched_ptr_s',
    'struct_hsa_queue_s', 'struct_hsa_region_s',
    'struct_hsa_signal_group_s', 'struct_hsa_signal_s',
    'struct_hsa_wavefront_s', 'uint16_t', 'uint32_t', 'uint64_t',
    'union_union_hsa_ext_amd_h_2329']
