script_help_msg = r"""
Usage:
- Run this script from blenders:

    blender --background --factory-startup -noaudio --python blender_stub_gen.py

Important!
To generate proper stub for bgl, do not use --background option
"""

try:
    import bpy  # Blender module
except ImportError:
    print("\nERROR: this script must run from inside Blender")
    print(script_help_msg)
    exit(1)

import math
import collections
import os
import inspect
import tokenize
import types
import typing
import logging

from io import StringIO
from pprint import pformat
from textwrap import indent
from collections import defaultdict
from typing import *

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s %(pathname)s:%(lineno)d  %(message)s')

__script_dir = os.path.dirname(__file__)
ROOT_DIR = os.path.join(__script_dir, 'bpy_stubs')


def is_named_tuple(value: Any) -> bool:
    return isinstance(value, tuple) and type(value).__name__ != 'tuple' and hasattr(value, '_fields')


C_STYLE_NAMED_TUPLE_FIELDS = {'n_fields', 'n_sequence_fields', 'n_unnamed_fields'}


def is_c_style_named_tuple(value: Any) -> bool:
    """ https://docs.python.org/3/c-api/sequence.html """
    return isinstance(value, tuple) and hasattr(value, 'n_sequence_fields') and hasattr(value, 'n_fields') and hasattr(
        value, 'n_unnamed_fields')


class InstanceMethodType:  # typing.Protocol - 3.8 needed
    def __func__(self, *args, **kwargs): pass


class Visitor:
    def __init__(self, module_name: str):
        self.module_name = module_name

    def visit_module(self, name: str, value: types.ModuleType):
        logging.error(
            f'element {self.module_name}.{name} is not supported ({self.module_name}.{name}: "{type(value)}" = {repr(value)})')

    def visit_builtin(self, name: str, value: Union[str, int, float, bool, bytes, bytearray, complex]):
        logging.error(
            f'element {self.module_name}.{name} is not supported ({self.module_name}.{name}: "{type(value)}" = {repr(value)})')

    def visit_mapping(self, name: str, value: collections.abc.Mapping):
        logging.error(
            f'element {self.module_name}.{name} is not supported ({self.module_name}.{name}: "{type(value)}" = {repr(value)})')

    def visit_sequence(self, name: str, value: collections.abc.Sequence):
        logging.error(
            f'element {self.module_name}.{name} is not supported ({self.module_name}.{name}: "{type(value)}" = {repr(value)})')

        if is_named_tuple(value):
            logging.error(f'type NamedTuple is not supported')

        if is_c_style_named_tuple(value):
            logging.error(f'c style NamedTuple is not supported')

    def visit_set(self, name: str, value: collections.abc.Set):
        logging.error(
            f'element {self.module_name}.{name} is not supported ({self.module_name}.{name}: "{type(value)}" = {repr(value)})')

    def visit_class_method_descriptor(self, name: str, value: types.ClassMethodDescriptorType):
        logging.error(
            f'element {self.module_name}.{name} is not supported ({self.module_name}.{name}: "{type(value)}" = {repr(value)})')

    def visit_member_descriptor(self, name: str, value: types.MemberDescriptorType):
        logging.error(
            f'element {self.module_name}.{name} is not supported ({self.module_name}.{name}: "{type(value)}" = {repr(value)})')

    def visit_method_descriptor(self, name: str, value: types.MethodDescriptorType):
        logging.error(
            f'element {self.module_name}.{name} is not supported ({self.module_name}.{name}: "{type(value)}" = {repr(value)})')

    def visit_wrapper_descriptor(self, name: str, value: types.WrapperDescriptorType):
        logging.error(
            f'element {self.module_name}.{name} is not supported ({self.module_name}.{name}: "{type(value)}" = {repr(value)})')

    def visit_get_set_descriptor(self, name: str, value: types.GetSetDescriptorType):
        logging.error(
            f'element {self.module_name}.{name} is not supported ({self.module_name}.{name}: "{type(value)}" = {repr(value)})')

    def visit_builtin_function(self, name: str, value: types.BuiltinFunctionType):
        logging.error(
            f'element {self.module_name}.{name} is not supported ({self.module_name}.{name}: "{type(value)}" = {repr(value)})')

    def visit_coroutine(self, name: str, value: types.CoroutineType):
        logging.error(
            f'element {self.module_name}.{name} is not supported ({self.module_name}.{name}: "{type(value)}" = {repr(value)})')

    def visit_async_generator(self, name: str, value: types.AsyncGeneratorType):
        logging.error(
            f'element {self.module_name}.{name} is not supported ({self.module_name}.{name}: "{type(value)}" = {repr(value)})')

    def visit_function(self, name: str, value: types.FunctionType):
        logging.error(
            f'element {self.module_name}.{name} is not supported ({self.module_name}.{name}: "{type(value)}" = {repr(value)})')

    def visit_code(self, name: str, value: types.CodeType):
        logging.error(
            f'element {self.module_name}.{name} is not supported ({self.module_name}.{name}: "{type(value)}" = {repr(value)})')

    def visit_instancemethod(self, name: str, value: InstanceMethodType):
        logging.error(
            f'element {self.module_name}.{name} is not supported ({self.module_name}.{name}: "{type(value)}" = {repr(value)})')

    def visit_class_definition(self, name: str, value: type):
        logging.error(
            f'element {self.module_name}.{name} is not supported ({self.module_name}.{name}: "{type(value)}" = {repr(value)})')
        if value.__name__ != name:
            # it is just an alias to class
            if value.__module__ in {self.module_name, None}:
                # it might be alias to private class
                logging.error('it is an alias to private class')
            else:
                logging.error('it is just an alias to class')
        else:
            pass

    def visit_variable(self, name: str, value: Any):
        """Visit variable with not obvious type: ether None or custom class"""
        logging.error(
            f'element {self.module_name}.{name} is not supported ({self.module_name}.{name}: "{type(value)}" = {repr(value)})')

        if inspect.isclass(type(value)) and type(value).__module__ in {self.module_name, None}:
            # it might variable with private class
            logging.error('it might be variable with private class')

    def fallback(self, name: str, value: Any):
        logging.error(
            f'element {self.module_name}.{name} is not supported ({self.module_name}.{name}: "{type(value)}" = {repr(value)})')

    def visit(self, attribute: str, value: Any):
        if isinstance(value, types.ModuleType):
            self.visit_module(attribute, value)

        elif isinstance(value, str):
            self.visit_builtin(attribute, value)

        elif isinstance(value, float):
            self.visit_builtin(attribute, value)

        elif isinstance(value, bytes):
            self.visit_builtin(attribute, value)

        elif isinstance(value, bytearray):
            self.visit_builtin(attribute, value)

        elif isinstance(value, int):
            self.visit_builtin(attribute, value)

        elif isinstance(value, complex):
            self.visit_builtin(attribute, value)

        elif isinstance(value, bool):
            self.visit_builtin(attribute, value)

        elif isinstance(value, collections.abc.Mapping):
            self.visit_mapping(attribute, value)

        elif isinstance(value, collections.abc.Sequence):
            self.visit_sequence(attribute, value)

        elif isinstance(value, collections.abc.Set):
            self.visit_set(attribute, value)

        elif isinstance(value, types.ClassMethodDescriptorType):
            self.visit_class_method_descriptor(attribute, value)

        elif isinstance(value, types.MemberDescriptorType):
            self.visit_member_descriptor(attribute, value)

        elif isinstance(value, types.MethodDescriptorType):
            self.visit_method_descriptor(attribute, value)

        elif isinstance(value, types.WrapperDescriptorType):
            self.visit_wrapper_descriptor(attribute, value)

        elif isinstance(value, types.GetSetDescriptorType):
            self.visit_get_set_descriptor(attribute, value)

        elif isinstance(value, types.BuiltinFunctionType):  # same as types.BuiltinMethodType
            self.visit_builtin_function(attribute, value)

        elif isinstance(value, types.CoroutineType):
            self.visit_coroutine(attribute, value)

        elif isinstance(value, types.AsyncGeneratorType):
            self.visit_async_generator(attribute, value)

        elif isinstance(value, types.FunctionType) or isinstance(value, types.MethodType):  # LambdaType == FunctionType
            self.visit_function(attribute, value)

        elif isinstance(value, types.CodeType):
            self.visit_code(attribute, value)

        elif hasattr(value, '__func__'):  # instancemethod type (not available in types module)
            self.visit_instancemethod(attribute, value)

        elif inspect.isclass(value):
            self.visit_class_definition(attribute, value)

        elif type(value) is not type and inspect.isclass(type(value)):
            # it is variable with custom class
            self.visit_variable(attribute, value)

        else:
            logging.fatal(f'unknown type: {self.module_name}.{attribute}: {type(value)}={repr(value)}')
            self.fallback(attribute, value)


class Item(NamedTuple):
    attribute: str
    value: Any


class Accumulator(Visitor):
    def __init__(self, module_name: str):
        super().__init__(module_name)
        self.accumulated: DefaultDict[type, List[Item]] = defaultdict(list)
        visit_methods_in_visitor = len([fn for fn in Visitor.__dict__.keys() if fn.startswith('visit_')])
        visit_methods_in_accumulator = len([fn for fn in Accumulator.__dict__.keys() if fn.startswith('visit_')])
        assert visit_methods_in_visitor == visit_methods_in_accumulator, \
            f'Sanity check failed. Accumulator should override all visit_* methods ({visit_methods_in_visitor} == {visit_methods_in_accumulator})'

    def visit_module(self, name: str, value: types.ModuleType):
        self.accumulated[types.ModuleType].append(Item(name, value))

    def visit_mapping(self, name: str, value: collections.abc.Mapping):
        self.accumulated[collections.abc.Mapping].append(Item(name, value))

    def visit_sequence(self, name: str, value: collections.abc.Sequence):
        self.accumulated[collections.abc.Sequence].append(Item(name, value))

    def visit_set(self, name: str, value: collections.abc.Set):
        self.accumulated[collections.abc.Set].append(Item(name, value))

    def visit_builtin(self, name: str, value: Union[str, int, float, bool, bytes, bytearray, complex]):
        self.accumulated['builtin'].append(Item(name, value))

    def visit_class_method_descriptor(self, name: str, value: types.ClassMethodDescriptorType):
        self.accumulated[types.ClassMethodDescriptorType].append(Item(name, value))

    def visit_member_descriptor(self, name: str, value: types.MemberDescriptorType):
        self.accumulated[types.MemberDescriptorType].append(Item(name, value))

    def visit_method_descriptor(self, name: str, value: types.MethodDescriptorType):
        self.accumulated[types.MethodDescriptorType].append(Item(name, value))

    def visit_wrapper_descriptor(self, name: str, value: types.WrapperDescriptorType):
        self.accumulated[types.WrapperDescriptorType].append(Item(name, value))

    def visit_get_set_descriptor(self, name: str, value: types.GetSetDescriptorType):
        self.accumulated[types.GetSetDescriptorType].append(Item(name, value))

    def visit_builtin_function(self, name: str, value: types.BuiltinFunctionType):
        self.accumulated[types.BuiltinFunctionType].append(Item(name, value))

    def visit_coroutine(self, name: str, value: types.CoroutineType):
        self.accumulated[types.CoroutineType].append(Item(name, value))

    def visit_async_generator(self, name: str, value: types.AsyncGeneratorType):
        self.accumulated[types.AsyncGeneratorType].append(Item(name, value))

    def visit_function(self, name: str, value: types.FunctionType):
        self.accumulated[types.FunctionType].append(Item(name, value))

    def visit_code(self, name: str, value: types.CodeType):
        self.accumulated[types.CodeType].append(Item(name, value))

    def visit_instancemethod(self, name: str, value: InstanceMethodType):
        # raise NotImplementedError
        self.accumulated["instancemethod"].append(Item(name, value))

    def visit_class_definition(self, name: str, value: type):
        self.accumulated[type].append(Item(name, value))

    def visit_variable(self, name: str, value: Any):
        # raise NotImplementedError
        self.accumulated["variable"].append(Item(name, value))

    def fallback(self, name: str, value: Any):
        self.accumulated["fallback"].append(Item(name, value))

    @staticmethod
    def _resolve_class_order_in_module(classes: List[Item], module: str) -> List[Item]:
        """Makes variant of DAG (Directed Acyclic Graph) topological sort to find order of classes

        Best to test on blender aud module

        TODO: this is not optimal implementation
        """
        sorted_classes = [item for item in sorted(classes, key=lambda item: item.attribute)]
        result = []
        ordering: Dict[Item, List[type]] = {}
        # map class to their bases (mro). Ignore object and self
        for item in sorted_classes:
            cls_mro = [mro for mro in inspect.getmro(item.value) if
                       inspect.getmodule(mro).__name__ in {module, 'builtins'} and mro not in {object, item.value}]
            ordering[item] = cls_mro
        min_len = 0  # prevents deadlock if not all classes are listed in input
        while ordering and min_len == 0:
            min_len = math.inf
            # logging.debug([i for i in ordering.keys()])
            for item, cls_mro in ordering.copy().items():
                # logging.debug(ordering)
                if cls_mro:
                    min_len = min(len(cls_mro), min_len)
                    continue
                min_len = 0
                result.append(item)
                del ordering[item]
                for k in ordering.keys():
                    try:
                        ordering[k].remove(item.value)
                        # logging.debug(f'Deleting in {k}')
                    except ValueError:
                        pass
        if min_len != 0:
            result.extend(item for item in ordering.keys())
        return result

    def replay(self, visitor: Visitor):
        """Preprocess self.accumulated and re-visit accumulated elements

        * Sort entries by name
        * Sort classes by resolution order
        """
        # process accumulated data about module
        for key in self.accumulated.keys():
            self.accumulated[key].sort(key=lambda item: item.attribute)
        self.accumulated[type] = self._resolve_class_order_in_module(self.accumulated[type], self.module_name)

        # replay in particular order
        key_order = [
            types.ModuleType,
            "builtin",
            "variable",
            collections.abc.Mapping,
            collections.abc.Sequence,
            collections.abc.Set,
            types.GetSetDescriptorType,
            types.WrapperDescriptorType,
            types.MemberDescriptorType,
            types.BuiltinFunctionType,
            types.CoroutineType,
            types.AsyncGeneratorType,
            types.FunctionType,
            types.ClassMethodDescriptorType,
            types.MethodDescriptorType,
            "instancemethod",
            type,
            types.CodeType,
            "fallback",
        ]
        assert len([fn for fn in Visitor.__dict__.keys() if fn.startswith('visit_')]) + 1 == len(key_order), \
            'Sanity check failed. Accumulator might skip some types during replay'
        for key in key_order:
            for item in self.accumulated[key]:
                visitor.visit(item.attribute, item.value)


def _is_inherited(attribute: str, cls: type):
    for base in inspect.getmro(cls)[1:-1]:  # ignore cls (first result) and object (last)
        # logging.debug(f'checking {attribute} of {cls.__name__} in {base}: {dir(base)}')
        if attribute in dir(base):
            return True
    return False


class StubGen(Visitor):
    def __init__(self, module_name: str, io: "TextIO"):
        super().__init__(module_name)
        self.io = io
        self.indent_level = 0
        self.imported_modules = set()

    def _indent_inc(self):
        self.indent_level += 4

    def _indent_dec(self):
        self.indent_level -= 4

    def _write(self, text: str):
        for line in text.splitlines(keepends=True):
            self.io.write(f'{self.indent_level * " "}{line}')

    def _write_constant(self, name: str, value: Any, *, value_type: type = None, value_type_as_str: str = None,
                        stub_value=False, add_doc=False) -> NoReturn:
        """

        :param name: attribute name
        :param value: attribute value
        :param value_type_as_str: use exactly this string as typehint. If None use value_type or as fallback infer type from value
        :param stub_value: write ellipsis (...) with typehint if true else just repr of value without typehint
        :param add_doc:
        """
        assert not (value_type and value_type_as_str)
        if not stub_value:
            self._write(f'{name} = {repr(value)}\n')
            if add_doc:
                self._write(f'"""{value.__doc__}\n"""\n')
            return

        if value_type_as_str:
            type_hint = value_type_as_str
        else:
            type_hint = value_type or self._get_type_hint(value)

        if type_hint:
            self._write(f'{name}: {type_hint} = ...\n')
        else:
            self._write(f'{name} = ...\n')
        if not add_doc:
            return
        self._write(f'"""Runtime value: {pformat(repr(value))}')
        # nobody cares about docs from built in types
        if value.__doc__ and type(value) not in {int, float, complex, bytes, bytearray, tuple, list, dict, str}:
            self._write('\n')
            self._write(value.__doc__)
        self._write(' """\n')

    def _get_type_hint(self, value: Any) -> str:
        value_type = type(value)
        module = value_type.__module__
        if value is None:
            type_hint = 'Optional'
            # logging.warning(f'{self.module_name}.{name} = None, can not get typehint')
        elif module not in {'builtins', self.module_name}:
            if module not in self.imported_modules and self.indent_level == 0:  # todo hack
                self._write(f'import {module}\n')
                self.imported_modules.add(module)
            type_hint = f'"{module}.{value_type.__name__}"'
        else:
            type_hint = f'"{value_type.__name__}"'
        return type_hint

    def visit_module(self, name: str, value: types.ModuleType):
        if value.__name__.startswith(self.module_name + '.'):
            self._write(f'from . import {name}\n')
        else:
            if value.__name__ != name:
                self._write(f'import {value.__name__} as {name}\n')
            else:
                self._write(f'import {name}\n')

    def visit_builtin(self, name: str, value: Union[str, int, float, bool, bytes, bytearray, complex]):
        self._write_constant(name, value)

    def visit_variable(self, name: str, value: Any):
        self._write_constant(name, value, stub_value=True)

        # if value and inspect.isclass(type(value)) and type(value).__module__ in {self.module_name, None, 'builtins'}:
        # it might variable with private class
        # logging.debug(f'{self.module_name}.{name}: {type(value)} private classes are not supported')

    def visit_sequence(self, name: str, value: collections.abc.Sequence):
        if is_named_tuple(value):
            # so far there was no need to support it
            logging.error(f'{self.module_name}.{name} type NamedTuple is not supported')
            self._write_constant(name, value, stub_value=True, add_doc=True)
            return

        if is_c_style_named_tuple(value):
            self._write(f'\nclass __{name}:  # Protocol\n')
            self._indent_inc()
            if value.__doc__:
                self._write(f'"""{value.__doc__}"""\n')
            for attr, value in inspect.getmembers(value):
                if attr.startswith('__') or attr in C_STYLE_NAMED_TUPLE_FIELDS or attr in {'count', 'index'}:
                    continue
                self.visit(attr, value)
            self._indent_dec()
            self._write('\n')
            self._write_constant(name, value, value_type_as_str=f'Union[__{name}, Tuple]', stub_value=True,
                                 add_doc=True)
            return

        # allow for printing simple types
        if all(type(tup) in {str, int, float, bool, bytes, bytearray, complex} for tup in value):
            self._write_constant(name, value)
        else:
            self._write_constant(name, value, stub_value=True, add_doc=True)

    def visit_set(self, name: str, value: collections.abc.Set):
        # set will be printed in random order...
        # todo not fully supported yet
        if type(value) is frozenset:
            items_str = ','.join(repr(item) for item in sorted(value))
            self._write(f'{name} = frozenset({{{items_str}}})\n')
        else:  # this is some kind of subclass of frozenset
            self._write_constant(name, value, stub_value=True, add_doc=True)

    def visit_mapping(self, name: str, value: collections.abc.Mapping):
        # todo not fully supported yet
        logging.warning(f'{self.module_name}.{name}: {type(value)} mapping is not fully supported')
        self._write_constant(name, value, stub_value=True)

    def visit_builtin_function(self, name: str, value: types.BuiltinFunctionType):
        try:
            signature = inspect.signature(value)
        except ValueError as e:
            # most likely: no signature found for builtin <built-in function {function_name}>
            signature = '(*args, **kwargs)'
            # logging.warning(f'function {self.module_name}.{name}: {e}')
        except RuntimeError as e:
            logging.error(f'something went very wrong with getting function signature: {e} '
                          f'Is {self.module_name}.{name}.__text_signature__ correct? (__text_signature__="{value.__text_signature__}")')
            return
        except tokenize.TokenError as e:
            logging.error(
                f'syntax error in {self.module_name}.{name}.__text_signature__="{value.__text_signature__}": {e}')
            return
        self._write(f'def {name}{signature}:\n')
        self._indent_inc()
        if value.__doc__:
            self._write(f'"""{value.__doc__} """\n')
        self._write(f'...\n\n')
        self._indent_dec()

    def visit_instancemethod(self, name: str, value: InstanceMethodType):
        # this is probably only alias to another function defined in c
        # but usually __module__ is not set and it is not possible to track source
        actual_function: types.FunctionType = value.__func__
        self.visit_builtin_function(name, actual_function)

    def visit_function(self, name: str, value: types.FunctionType):
        self.visit_builtin_function(name, value)

    def visit_class_definition(self, name: str, value: type):
        if value.__module__ == 'builtins' and value.__name__ == name:
            # todo this is correct class definition (like bpy_struct), but we miss __module__ information
            pass
        elif value.__module__ != self.module_name or value.__name__ != name:
            # it is just an alias to class
            if value.__module__ not in self.imported_modules:
                self.imported_modules.add(value.__module__)
                self._write(f'import {value.__module__}\n')
            self._write(f'{name} = {value.__module__}.{value.__name__}\n')
            return

        bases = []
        for base in value.__bases__:
            if base.__module__ and base.__module__ != self.module_name:
                if base.__module__ != 'builtins' and base.__module__ != self.module_name:
                    bases.append(f'{base.__module__}.{base.__name__}')
                else:
                    # logging.warning(f'Adding class that is defined as builtin: {base.__module__}.{base.__name__}')
                    bases.append(base.__name__)
            else:
                bases.append(base.__name__)

        # there is always `object` in bases
        self._write(f'class {name}({",".join(bases)}):\n')

        self._indent_inc()
        if value.__doc__:
            self._write(f'"""{value.__doc__}\n"""\n\n')

        # logging.debug(f'{self.module_name}.{value.__name__} = {value}')
        acc = Accumulator(self.module_name)  # , value.__name__, inspect.getmro(value))
        members = inspect.getmembers(value)
        function_like_types = {
            types.BuiltinFunctionType, types.FunctionType, types.MethodType,
            types.MethodDescriptorType, types.WrapperDescriptorType, types.BuiltinMethodType,
            types.GetSetDescriptorType}
        for attribute, attr_value in members:
            if attribute.startswith('__'):
                continue
            # skip inherited function-like attributes
            if _is_inherited(attribute, cls=value) and type(attr_value) in function_like_types:
                # logging.debug(f'skipped (is inherited): {name}.{attribute} = {attr_value}')
                continue
            acc.visit(attribute, attr_value)

        acc.replay(self)
        # it is possible for replay to write nothing. Always add ...
        self._write('...\n\n')
        self._indent_dec()

    def visit_method_descriptor(self, name: str, value: types.MethodDescriptorType):
        # todo not tested
        self.visit_builtin_function(name, value)

    def visit_class_method_descriptor(self, name: str, value: types.ClassMethodDescriptorType):
        # todo not tested
        self._write('@classmethod\n')
        self.visit_builtin_function(name, value)

    def visit_get_set_descriptor(self, name: str, value: types.GetSetDescriptorType):
        # logging.debug(f'{self.module_name}.{name}s={value}, {value.__objclass__}')
        # todo can we do better?
        self._write(f'{name} = property(lambda self: object(), lambda self, v: None, lambda self: None)\n')
        if value.__doc__:
            self._write(f'"""{value.__doc__}"""\n')


class BlenderStubGen(StubGen):
    def visit(self, attribute: str, value: Any) -> NoReturn:
        import bpy, mathutils
        module_prefix = 'bpy.types.' if self.module_name != 'bpy.types' else ''

        # bl_rna, rna_type are blender quirk, that is does not have expected fields, when you follow type information
        if attribute in {'bl_rna', 'rna_type'}:
            self._write_constant(name=attribute, value=value, stub_value=True)

        elif isinstance(value, bpy.types.Property):
            self.visit_bpy_property(attribute, value)

        elif isinstance(value, bpy.types.PropertyGroup):  # this is not subclass of Property
            # todo not tested
            self._write_constant(name=attribute, value=value, stub_value=True, add_doc=True)

        elif isinstance(value, bpy.types.Function):
            self.visit_bpy_function(value)

        # the following types are defined as subclasses of RNA_Types class (bpy.types) and can not be properly hinted
        elif type(value) == bpy.types.bpy_func:
            self.visit_builtin_function(attribute, value)
            raise Exception

        elif type(value) == bpy.types.bpy_prop:
            # todo not tested
            self._write_constant(name=attribute, value=value, value_type_as_str=f'{module_prefix}bpy_prop',
                                 stub_value=True, add_doc=True)

        elif type(value) == bpy.types.bpy_prop_array:
            # todo not tested
            self._write_constant(name=attribute, value=value, value_type_as_str=f'{module_prefix}bpy_prop_array',
                                 stub_value=True, add_doc=True)

        elif type(value) == bpy.types.bpy_prop_collection:
            # todo corner case: bpy.data.shape_keys is a collection,
            # but the information about fixed_type is in bpy.types.BlendData.shape_keys or
            # bpy.types.BlendData.bl_rna.properties['shape_keys'].fixed_type
            self._write_constant(name=attribute, value=value,
                                 value_type_as_str=f'{module_prefix}{value.rna_type.bl_rna.identifier}',
                                 stub_value=True)

        elif inspect.isclass(value) and issubclass(value, bpy.types.bpy_struct):
            self.visit_bpy_struct_definition(name=attribute, value=value)

        elif type(value) == bpy.types.bpy_struct:
            self._write_constant(name=attribute, value=value, value_type_as_str=f'{module_prefix}bpy_struct',
                                 stub_value=True)

        elif type(value) == bpy.types.bpy_struct_meta_idprop:
            self._write_constant(name=attribute, value=value,
                                 value_type_as_str=f'{module_prefix}bpy_struct_meta_idprop',
                                 stub_value=True)
            # logging.warning(f'That type is so weird that I will pass: {attribute}={value}')

        elif isinstance(value, mathutils.Vector):
            raise NotImplementedError()
        else:
            super().visit(attribute, value)

    def visit_bpy_function(self, value: bpy.types.Function):
        if not value.use_self:
            if value.use_self_type:
                self._write('@classmethod\n')
            else:
                self._write('@staticmethod\n')
        self._write(f'def {value.identifier}(')
        # todo what is: value.use_self ??
        if value.use_self:
            self.io.write('self')
        if value.use_self_type:
            self.io.write('cls')
        if (value.use_self or value.use_self_type) and len(value.parameters) != 0:
            self.io.write(', ')
        ret = None
        for i, param in enumerate(sorted(value.parameters, key=lambda prop: (not prop.is_required, prop.identifier))):
            param: Type[bpy.types.Property]
            if param.is_output:
                ret = param
                continue
            # todo I do not know what to do with these:
            if param.is_argument_optional:
                pass
            if param.is_required:
                pass

            if isinstance(param, bpy.types.PointerProperty):
                if param.is_required:
                    self.io.write(f'{param.identifier}')
                else:
                    self.io.write(f'{param.identifier} = None')
            else:
                if param.is_required:
                    self.io.write(f'{param.identifier}')
                else:
                    self.io.write(f'{param.identifier} = {repr(param.default)}')
            if i != len(value.parameters) - 1:
                self.io.write(', ')
        if ret:
            rtype = getattr(param, 'fixed_type', param)
            rtype = type(rtype)
            if rtype.__module__ == self.module_name:
                self.io.write(f') -> {rtype.__name__}:\n')
            else:
                self.io.write(f') -> {rtype.__module__}.{rtype.__name__}:\n')
        else:
            self.io.write(f') -> NoReturn:\n')
        self._indent_inc()
        self._write(f'"""{value.description}')
        if len(value.parameters) != 0:
            self.io.write('\n\n')
        for param in value.parameters:
            if param.is_output:
                continue
            doc = self.get_bpy_property_doc(param)
            self._write(f':param {param.identifier}: {doc}\n')
        if ret and ret.description:
            self._write(f':returns: {ret.description}\n')
        self._write('"""\n')
        self._write('...\n')
        self._indent_dec()

    def visit_bpy_property(self, attribute, value: bpy.types.Property) -> NoReturn:
        module_prefix = 'bpy.types.' if self.module_name != 'bpy.types' else ''
        property_type = f'"{module_prefix}{type(value).__name__}"'
        doc = ''
        if type(value) == bpy.types.Property:
            self._write_constant(attribute, value, value_type_as_str=property_type, stub_value=True)

        elif isinstance(value, (bpy.types.IntProperty, bpy.types.FloatProperty)):
            self._write_constant(attribute, value, value_type_as_str=property_type, stub_value=True)
            doc = self.get_int_float_property_doc(value)

        elif isinstance(value, bpy.types.BoolProperty):
            self._write_constant(attribute, value, value_type_as_str=property_type, stub_value=True)
            doc = self.get_bpy_bool_property_doc(value)

        elif isinstance(value, bpy.types.StringProperty):
            self._write_constant(attribute, value, value_type_as_str=property_type, stub_value=True)
            doc = self.get_bpy_string_property_doc(value)

        elif isinstance(value, bpy.types.EnumProperty):
            # logging.debug(f'{self.module_name} {attribute}={value}')
            # if value.is_enum_flag:
            #     type_hint = f'Union[{property_type}, Set[str]]'
            # else:
            #     type_hint = f'Union[{property_type}, str]'
            self._write_constant(name=attribute, value=value, value_type_as_str=None, stub_value=True)
            # todo docstring becomes large when listing icons
            doc = self.get_bpy_enum_protperty_doc(value)
            # doc = f'Possible values (default={repr(value.default)})'

        elif isinstance(value, bpy.types.PointerProperty):
            self._write_constant(attribute, value, value_type_as_str=property_type, stub_value=True)
            doc = self.get_bpy_pointer_protperty_doc(value)

        elif isinstance(value, bpy.types.CollectionProperty):
            self._write_constant(attribute, value, value_type_as_str=property_type, stub_value=True)
            doc = self.get_bpy_pointer_protperty_doc(value)

        else:
            raise Exception(f'Unknown property: {attribute}={value}')

        if doc:
            self._write(f'"""{doc}"""\n')

    def get_bpy_pointer_protperty_doc(self,
                                      value: Union[bpy.types.CollectionProperty, bpy.types.PointerProperty]) -> str:
        doc = f'{value.name}'
        if value.is_readonly:
            doc += ' (readonly)'
        doc += '\n\n'
        if value.description:
            doc += f'{value.description}\n\n'
        doc += f':type {value.identifier}: {self._get_bpy_type_hint(value, allow_literal=True, allow_invalid_bpy_syntax=True)}\n\n'
        # todo we can probably create valid link with syntax :class:`bpy.types.NodeLink`
        # doc += f':type {value.identifier}: {type(value).__module__}.{type(value).__name__}[{type(value.fixed_type).__module__}.{type(value.fixed_type).__name__}]'
        if value.subtype != 'NONE':
            doc += f'\n* subtype={value.subtype}'
        if value.icon != 'NONE':
            doc += f'\n* icon={value.icon}'
        doc += '\n'
        return doc

    def get_int_float_property_doc(self, value: Union[bpy.types.FloatProperty, bpy.types.IntProperty]) -> str:
        doc = f'{value.name}'
        if value.is_readonly:
            doc += ' (readonly)'
        doc += '\n\n'
        if value.description:
            doc += f'{value.description}\n\n'
        doc += f':type {value.identifier}: {self._get_bpy_type_hint(value, allow_literal=True, allow_invalid_bpy_syntax=True)}\n\n'
        if value.array_length != 0:
            doc += f'* default_array={repr(tuple(value.default_array))}\n' \
                   f'* array_length={value.array_length}\n'
        else:
            doc += f'* default={repr(value.default)}\n'
        doc += f'* hard_min={value.hard_min}\n' \
               f'* hard_max={value.hard_max}'
        if value.unit != 'NONE':
            doc += f'\n* unit={value.unit}'
        if value.subtype != 'NONE':
            doc += f'\n* subtype={value.subtype}'
        if value.icon != 'NONE':
            doc += f'\n* icon={value.icon}'
        doc += '\n'
        return doc

    def get_bpy_bool_property_doc(self, value: bpy.types.BoolProperty) -> str:
        doc = f'{value.name}'
        if value.is_readonly:
            doc += ' (readonly)'
        doc += '\n\n'
        if value.description:
            doc += f'{value.description}\n\n'
        doc += f':type {value.identifier}: {self._get_bpy_type_hint(value, allow_literal=True, allow_invalid_bpy_syntax=True)}\n\n'
        if value.array_length != 0:
            doc += f'* default_array={repr(tuple(value.default_array))}\n' \
                   f'* array_length={value.array_length}'
        else:
            doc += f'* default={repr(value.default)}'
        if value.subtype != 'NONE':
            doc += f'\n* subtype={value.subtype}'
        if value.icon != 'NONE':
            doc += f'\n* icon={value.icon}'
        doc += '\n'
        return doc

    def get_bpy_string_property_doc(self, value: bpy.types.StringProperty) -> str:
        doc = f'{value.name}'
        if value.is_readonly:
            doc += ' (readonly)'
        doc += '\n\n'
        if value.description:
            doc += f'{value.description}\n\n'

        doc += f':type {value.identifier}: {self._get_bpy_type_hint(value, allow_literal=True, allow_invalid_bpy_syntax=True)}\n\n'
        doc += f'* default={repr(value.default)}\n' \
               f'* length_max={value.length_max}'
        if value.subtype != 'NONE':
            doc += f'\n* subtype={value.subtype}'
        if value.icon != 'NONE':
            doc += f'\n* icon={value.icon}'
        doc += '\n'
        return doc

    def get_bpy_enum_protperty_doc(self, value: bpy.types.EnumProperty):
        doc = f'{value.name}'
        if value.is_readonly:
            doc += ' (readonly)'
        doc += '\n\n'
        if value.description:
            doc += f'{value.description}\n\n'
        doc += f':type {value.identifier}: {self._get_bpy_type_hint(value, allow_literal=True, allow_invalid_bpy_syntax=True)}\n\n'
        doc += f'* default={repr(value.default)}\n'
        if value.subtype != 'NONE':
            doc += f'\n* subtype={value.subtype}'
        if value.icon != 'NONE':
            doc += f'\n* icon={value.icon}'
        doc += f'\n* Possible Items: '
        # make listing shorter:
        doc += ', '.join(item.identifier for item in value.enum_items if
                         item.identifier == item.name and not item.description)
        doc += '\n'
        for item in value.enum_items:
            item: bpy.types.EnumPropertyItem
            if item.identifier == item.name and not item.description:
                continue
            if item.identifier != item.name:
                doc += f'    * {item.identifier} - {item.name}'
            else:
                doc += f'    * {item.identifier}'

            if item.description:
                doc += f' - {item.description}\n'
            else:
                doc += '\n'
        doc += '\n'
        return doc

    def visit_bpy_struct_definition(self, name: str, value: type) -> NoReturn:
        # logging.debug(f'{name}={value}')
        if value.__module__ == 'builtins' and value.__name__ == name:
            # todo this is correct class definition (like bpy_struct)
            pass
        elif value.__module__ != self.module_name or value.__name__ != name:
            # it is just an alias to class
            if value.__module__ not in self.imported_modules:
                self.imported_modules.add(value.__module__)
                self._write(f'import {value.__module__}\n')
            self._write(f'{name} = {value.__module__}.{value.__name__}\n')
            return

        bases = []
        for base in value.__bases__:
            if base.__module__ and base.__module__ != self.module_name:
                if base.__module__ != 'builtins' and base.__module__ != self.module_name:
                    bases.append(f'{base.__module__}.{base.__name__}')
                else:
                    # logging.warning(f'Adding class that is defined as builtin: {base.__module__}.{base.__name__}')
                    bases.append(base.__name__)
            else:
                bases.append(base.__name__)

        # there is always `object` in bases
        self._write(f'class {name}({",".join(bases)}):\n')

        self._indent_inc()
        bl_rna = getattr(value, 'bl_rna', None)
        if bl_rna or value.__doc__:
            self._write(f'"""{bl_rna.name} ({bl_rna.__module__}.{bl_rna.identifier})\n\n')
            if value.__doc__:
                self._write(f'{value.__doc__}\n\n')
            self._write(f'{bl_rna.description}"""\n\n')

        members = {m[0]: m[1] for m in inspect.getmembers(value)}
        if bl_rna:
            properties: bpy.types.bpy_prop_collection = getattr(bl_rna, 'properties', None)
            if properties:
                for prop in properties:
                    prop: bpy.types.Property
                    # watch out, duplicates possible, but we prefer Property
                    members.update({prop.identifier: prop})
            functions = getattr(bl_rna, 'functions', None)
            if functions:
                for fn in functions:
                    fn: bpy.types.Function
                    # watch out, duplicates possible, but we prefer Function
                    members.update({fn.identifier: fn})
        acc = Accumulator(self.module_name)  # , value.__name__, inspect.getmro(value))
        function_like_types = {
            types.BuiltinFunctionType, types.FunctionType, types.MethodType, types.MethodDescriptorType,
            types.WrapperDescriptorType, types.BuiltinMethodType, types.GetSetDescriptorType}
        for attribute, attr_value in members.items():
            if attribute.startswith('__'):
                continue
            # skip inherited function-like attributes
            if _is_inherited(attribute, cls=value) and type(attr_value) in function_like_types:
                # logging.debug(f'skipped (is inherited): {name}.{attribute} = {attr_value}')
                continue
            acc.visit(attribute, attr_value)

        acc.replay(self)
        # it is possible for replay to write nothing. Always add ...
        self._write('...\n\n')
        self._indent_dec()

    def visit_bpy_operator(self, name: str, value: Union[bpy.ops._BPyOpsSubModOp, bpy.types.OperatorProperties],
                           allow_literal=False):
        """visit_bpy_operator is not part of main self.visit loop, because it is incredibly difficult to discover operator reliably
        Writes operator in style:

        >>> @overload
        ... def easing_type(context_copy: Dict[str, Any], C_exec: str, C_undo: int, **kwargs) -> Set[str]:
        ...    ...
        ...
        ... @overload
        ... def easing_type(*, rna_type = None, type = 'AUTO') -> Set[str]:
        ...     ...
        """
        rna_type = value.get_rna_type()
        assert len(rna_type.functions) == 0, 'Operators usually do not have functions'

        # write typed dictionaries used for type hints and documentation
        refined_hints = {}
        for prop in rna_type.properties:
            if prop.identifier in {'rna_type', 'bl_rna'}:
                continue
            if isinstance(prop, (bpy.types.PointerProperty, bpy.types.CollectionProperty)):
                is_accessible_in_bpy_types = getattr(bpy.types, prop.fixed_type.identifier, None)
                self._write(f'class __{prop.identifier}(TypedDict):  # python 3.8!!\n')
                self._indent_inc()
                if is_accessible_in_bpy_types:
                    self._write(f'"""This class is used only for documentation. '
                                f'True type is {prop.fixed_type.__module__}.{prop.fixed_type.identifier}"""\n')
                else:
                    self._write(f'"""This class is used only for documentation"""\n')
                for _prop in prop.fixed_type.properties:
                    if _prop.identifier in {'rna_type', 'bl_rna'}:
                        continue
                    self._write(
                        f'{_prop.identifier}: {self._get_bpy_type_hint(_prop, allow_literal=allow_literal)}\n')
                    doc = self.get_bpy_property_doc(_prop)
                    self._write(f'"""{_prop.identifier}: {doc}"""\n')
                self._write('...\n\n')
                self._indent_dec()
            if isinstance(prop, bpy.types.PointerProperty):
                # it is possible that prop.fixed_type is not exposed in bpy.types
                refined_hints[prop.identifier] = f'__{prop.identifier}'
            elif isinstance(prop, bpy.types.CollectionProperty):
                refined_hints[prop.identifier] = f'Sequence[__{prop.identifier}]'

        _literal = 'Literal["RUNNING_MODAL", "CANCELLED", "FINISHED", "PASS_THROUGH"]'
        _str = 'str'
        self._write(f'@overload\n'
                    f'def {name}(context_copy: Dict[str, Any] = ..., C_exec: str = ..., C_undo: int = ..., **kwargs) -> Set[{_literal if allow_literal else _str}]:\n'
                    f'    ...\n\n')

        self._write(f'@overload\n'
                    f'def {name}(')

        # write function arguments
        if len(rna_type.properties) > 1:
            self.io.write('*, ')
        for i, param in enumerate(
            sorted(rna_type.properties, key=lambda prop: (not prop.is_required, prop.identifier))):
            if param.identifier == 'rna_type':
                continue
            if isinstance(param, bpy.types.PointerProperty):
                if param.is_required:
                    self.io.write(f'{param.identifier}: {refined_hints[param.identifier]}')
                else:
                    self.io.write(f'{param.identifier}: {refined_hints[param.identifier]} = None')
            elif isinstance(param, bpy.types.CollectionProperty):
                # example: bpy.ops.clip.open(directory="/path", files=[{"name":"Bez nazwy.png"}], relative_path=True)
                if param.is_required:
                    self.io.write(f'{param.identifier}: {refined_hints[param.identifier]}')
                else:
                    self.io.write(f'{param.identifier}: {refined_hints[param.identifier]} = None')
            else:
                if param.is_required:
                    self.io.write(f'{param.identifier}')
                else:
                    if hasattr(param, 'default_array') and param.default_array:
                        self.io.write(f'{param.identifier} = {repr(tuple(param.default_array))}')
                    else:
                        self.io.write(f'{param.identifier} = {repr(param.default)}')
            if i != len(rna_type.properties) - 1:
                self.io.write(', ')
        if allow_literal:
            self.io.write(') -> Set[Literal["RUNNING_MODAL", "CANCELLED", "FINISHED", "PASS_THROUGH"]]:\n')
        else:
            self.io.write(') -> Set[str]:\n')

        # write docstring
        self._indent_inc()
        self._write(f'"""{value.__doc__}\n\n'
                    f'* bl_options={value.bl_options}\n'
                    f'* idname={value.idname()}\n\n')
        for param in rna_type.properties:
            if param.identifier == 'rna_type':
                continue
            doc = self.get_bpy_property_doc(param)
            self._write(f':param {param.identifier}: {doc}\n')
        self._write('"""\n')
        self._write('...\n\n')
        self._indent_dec()

    def _get_bpy_type_hint(self, value: Union[Any, bpy.types.Property], allow_invalid_bpy_syntax=False,
                           allow_literal=False) -> str:
        if isinstance(value, bpy.types.Property):
            if isinstance(value, bpy.types.PointerProperty):
                # return self._get_type_hint(value.fixed_type)
                if self.module_name != 'bpy.types' and value.fixed_type == 'bpy.types':
                    if allow_invalid_bpy_syntax:
                        return f'bpy.types.PointerProperty[{value.fixed_type.__module__}.{value.fixed_type.identifier}]'
                    else:
                        return f'bpy.types.PointerProperty[{value.fixed_type.__module__}.{value.fixed_type.identifier}]'
                else:
                    if allow_invalid_bpy_syntax:
                        return f'PointerProperty[{value.fixed_type.identifier}]'
                    else:
                        return value.fixed_type.identifier
            elif isinstance(value, bpy.types.CollectionProperty):
                # return self._get_type_hint(value.fixed_type)
                if self.module_name != 'bpy.types':
                    if allow_invalid_bpy_syntax:
                        return f'bpy.types.CollectionProperty[{value.fixed_type.__module__}.{value.fixed_type.identifier}]'
                    else:
                        return f'Collection[{value.fixed_type.__module__}.{value.fixed_type.identifier}]'  # typing.Collection
                else:
                    if allow_invalid_bpy_syntax:
                        return f'CollectionProperty[{value.fixed_type.identifier}]'
                    else:
                        return f'Collection[{value.fixed_type.identifier}]'  # typing.Collection, not ideal
            elif isinstance(value, bpy.types.StringProperty):
                return 'str'
            elif isinstance(value, bpy.types.IntProperty):
                if value.array_length == 0:
                    return 'int'
                inner = ', '.join('int' for _ in range(value.array_length))
                return f'Tuple[{inner}]'  # todo not actually a tuple, but this is the closest to what we need
            elif isinstance(value, bpy.types.FloatProperty):
                if value.array_length == 0:
                    return 'float'
                inner = ', '.join('float' for _ in range(value.array_length))
                return f'Tuple[{inner}]'
            elif isinstance(value, bpy.types.BoolProperty):
                if value.array_length == 0:
                    return 'bool'
                inner = ', '.join('bool' for _ in range(value.array_length))
                return f'Tuple[{inner}]'
            elif isinstance(value, bpy.types.EnumProperty):
                if value.is_enum_flag:
                    if allow_literal:
                        return 'Set[Literal[' + ', '.join(f'"{item.identifier}"' for item in value.enum_items) + ']]'
                    else:
                        return 'Set[str]'
                else:
                    if allow_literal:
                        return 'Literal[' + ', '.join(f'"{item.identifier}"' for item in value.enum_items) + ']'
                    else:
                        return 'str'
        else:
            return self._get_type_hint(value)
        return 'Any'

    def get_bpy_property_doc(self, param: bpy.types.Property) -> str:
        if isinstance(param, (bpy.types.IntProperty, bpy.types.FloatProperty)):
            return self.get_int_float_property_doc(param)
        elif isinstance(param, bpy.types.BoolProperty):
            return self.get_bpy_bool_property_doc(param)
        elif isinstance(param, bpy.types.StringProperty):
            return self.get_bpy_string_property_doc(param)
        elif isinstance(param, bpy.types.EnumProperty):
            return self.get_bpy_enum_protperty_doc(param)
        elif isinstance(param, bpy.types.PointerProperty):
            return self.get_bpy_pointer_protperty_doc(param)
        elif isinstance(param, bpy.types.CollectionProperty):
            return self.get_bpy_pointer_protperty_doc(param)
        else:
            raise Exception(f'Unknown parameter type {param}')


def pymodule2stub(basepath: str, module_file_name: str, module: types.ModuleType, skip_attributes: Set[str] = None,
                  additional_imports: Iterable[str] = None, additional_raw_imports: Iterable[str] = None):
    module2stub(basepath, module_file_name, module_name=module.__name__,
                module_members=inspect.getmembers(module),
                module_doc=module.__doc__,
                skip_attributes=skip_attributes,
                additional_imports=additional_imports,
                additional_raw_imports=additional_raw_imports)


def module2stub(basepath: str, module_file_name: str, module_name: str,
                module_members: List[Tuple[str, Any]],
                module_doc: str = None,
                skip_attributes: Set[str] = None,
                additional_imports: Iterable[str] = None,
                additional_raw_imports: Iterable[str] = None):
    skip_attributes = skip_attributes or set()
    additional_imports = additional_imports or set()
    additional_raw_imports = additional_raw_imports or set()
    filepath = os.path.join(basepath, f'{module_file_name}.pyi')

    file = open(filepath, 'w', encoding="utf-8")

    # The description of this module:
    if module_doc:
        file.write(f'"""{module_doc}\n"""\n\n')
    file.write('from typing import *\n')
    for imp in sorted(additional_imports):
        file.write(f'import {imp}\n')
    for imp in sorted(additional_raw_imports):
        file.write(f'{imp}\n')
    file.write('\n')

    accumulator = Accumulator(module_name=module_name)
    for attribute, value in sorted(module_members, key=lambda k: k[0]):
        if attribute.startswith("__"):
            continue
        if attribute in skip_attributes:
            continue

        accumulator.visit(attribute, value)
    visitor = BlenderStubGen(module_name, file)
    accumulator.replay(visitor)

    file.close()


def _discover_submodules(module: types.ModuleType) -> Set[types.ModuleType]:
    result = set()
    for attribute, value in inspect.getmembers(module):
        if isinstance(value, types.ModuleType):
            result.add(value)
    return result


def pypackage2stub(basepath: str, module: types.ModuleType):
    submodules = _discover_submodules(module)

    if not submodules:
        pymodule2stub(basepath, module.__name__, module)
        return

    package_basepath = os.path.join(basepath, module.__name__)
    os.makedirs(package_basepath, exist_ok=True)

    pymodule2stub(package_basepath, '__init__', module)

    for submodule in sorted(submodules, key=lambda mod: mod.__name__):
        submodule_name = submodule.__name__

        logging.info(f'Generating {submodule_name}')
        if submodule_name.startswith(module.__name__ + '.'):
            submodule_name = submodule_name[len(module.__name__ + '.'):]
        pymodule2stub(package_basepath, submodule_name, submodule)
        assert not _discover_submodules(submodule), \
            f'recursive packages are not supported: {_discover_submodules(submodule)}. There was simply no need to do it'


def stub_bpy():
    import _bpy
    bpy_basedir = os.path.join(ROOT_DIR, '_bpy')
    os.makedirs(bpy_basedir, exist_ok=True)
    # todo stub bpy.context
    logging.info('Generate _bpy')
    pymodule2stub(basepath=bpy_basedir, module_file_name='__init__', module=_bpy,
                  skip_attributes={'app', 'types', 'context'},
                  additional_raw_imports={'from . import types', 'from . import app'})

    logging.info('Generate _bpy.props')
    pymodule2stub(bpy_basedir, 'props', _bpy.props)

    logging.info('Generate _bpy.ops')
    _stub_bpy_ops(bpy_basedir)

    # bpy.types is actually class, but we want to gather all possible information
    members = inspect.getmembers(type(_bpy.types))  # bpy_struct, bpy_prop, ... - only core types
    members.extend(inspect.getmembers(_bpy.types))  # *_OT_*, *_PT_*, *_MT_* and many more
    logging.info(f'Generate _bpy.types ({len(members)} elements)')

    module2stub(
        basepath=bpy_basedir, module_file_name='types', module_name=_bpy.types.__name__,
        module_members=members, module_doc=_bpy.types.__doc__)

    logging.info('Generate _bpy.app')
    app_basepath = os.path.join(bpy_basedir, 'app')
    os.makedirs(app_basepath, exist_ok=True)
    # attributes icons and timers are skipped because they are documented as separate modules
    module2stub(
        app_basepath, '__init__', module_name='app', module_doc=bpy.app.__doc__,
        module_members=inspect.getmembers(_bpy.app),
        skip_attributes={'icons', 'timers'}.union(C_STYLE_NAMED_TUPLE_FIELDS))

    # todo bpy.app.translations is misformatted
    logging.info('Generate bpy.app._translations_type')
    module2stub(
        app_basepath, '_translations_type', module_name='_translations_type', module_doc=_bpy.app.translations.__doc__,
        module_members=inspect.getmembers(_bpy.app.translations),
        skip_attributes=C_STYLE_NAMED_TUPLE_FIELDS)

    logging.info('Generate _bpy.app.icons')
    pymodule2stub(app_basepath, 'icons', _bpy.app.icons)

    logging.info('Generate _bpy.app.timers')
    pymodule2stub(app_basepath, 'timers', _bpy.app.timers)

    logging.info('Generate _bpy._utils_previews')
    pymodule2stub(bpy_basedir, '_utils_previews', _bpy._utils_previews)

    logging.info('Generate _bpy._utils_units')
    pymodule2stub(bpy_basedir, '_utils_units', _bpy._utils_units)

    logging.info('Generate _bpy.msgbus')
    pymodule2stub(bpy_basedir, 'msgbus', _bpy.msgbus)


def _stub_bpy_ops(bpy_basedir):
    ops_basedir = os.path.join(bpy_basedir, 'ops')
    os.makedirs(ops_basedir, exist_ok=True)

    import _bpy

    # bpy.ops overrides __dir__, we can safely get submodules
    submodules = []
    for submodule_name, submodule in inspect.getmembers(bpy.ops):
        submodules.append(submodule_name)
        file = open(os.path.join(ops_basedir, f'{submodule_name}.pyi'), 'w', encoding="utf-8")
        file.write('import bpy\n'
                   'from typing import *\n\n')
        visitor = BlenderStubGen(module_name=submodule_name, io=file)
        for op_name, op in sorted(inspect.getmembers(submodule), key=lambda op: op[0]):
            rna_type = op.get_rna_type()
            # some operators are already defined in bpy.types
            # _op_name = op.idname_py().split('.')[1]
            if getattr(bpy.types, rna_type.identifier, None):
                visitor.visit(attribute=op_name, value=rna_type)
            else:
                visitor.visit_bpy_operator(name=op_name, value=op, allow_literal=False)
        file.close()

    module2stub(
        basepath=ops_basedir, module_file_name='__init__', module_name='ops',
        module_members=inspect.getmembers(_bpy.ops),
        additional_raw_imports=set(f'from . import {mod}' for mod in submodules))


def _stub_bpy_ops_v2(bpy_basedir):
    '''Write blender operators in style (currently unused):

    >>> class layer_prev(bpy.ops._BPyOpsSubModOp):  # fake class definition, real value: <function bpy.ops.action.layer_prev at 0x1c6c127a218'>
    ...     """bpy.ops.action.layer_prev()
    ...     Switch to editing action in animation layer below the current action in the NLA Stack"""
    ...     bl_options = {'REGISTER', 'UNDO'}
    ...     idname = lambda : 'ACTION_OT_layer_prev'
    ...     get_rna_type: "_bpy.types.OperatorProperties" = ...
    ...     """Runtime value: <bpy_struct, Struct("ACTION_OT_layer_prev") at 0x000001C6BE102838>"""
    ...     def __init__(self, C_dict: Dict[str, Any] = ..., C_exec: str = ..., C_undo: int = ..., /) -> __OperatorResult:
    ...         """bpy.ops.action.layer_prev()"""
    '''
    import _bpy
    ops_basedir = os.path.join(bpy_basedir, 'ops')
    os.makedirs(ops_basedir, exist_ok=True)
    ops_submodules: Dict[str, List[str]] = defaultdict(list)
    for id_name in _bpy.ops.dir():
        id_split = id_name.split("_OT_", 1)
        if len(id_split) == 2:
            mod = id_split[0].lower()
            ops_submodules[mod].append(id_split[1])
        else:
            assert False, f'Unexpected operator formatting: {id_name}'
    for module, functions in sorted(ops_submodules.items(), key=lambda k: k[0]):
        # functions are instances of bpy.ops.
        file = open(os.path.join(ops_basedir, f'{module}.pyi'), 'w', encoding="utf-8")
        # visitor = StubGen(module, file)
        file.write('import bpy\n'
                   'import _bpy\n'
                   'from typing import *\n\n'
                   '__OperatorResult = Set[Literal["RUNNING_MODAL", "CANCELLED", "FINISHED", "PASS_THROUGH"]]\n\n')
        ops_imported = {'bpy'}

        for fn_name in sorted(functions):
            fn: bpy.ops._BPyOpsSubModOp = bpy.ops._bpy_ops_submodule__getattr__(module, fn_name)
            defined_type = getattr(bpy.types, fn.idname(), None)
            if defined_type:
                # some operators come from addons or bl_operators module and all we need to do is point to
                # implementation
                if defined_type.__module__ not in ops_imported:
                    file.write(f"import {defined_type.__module__}\n")
                    ops_imported.add(defined_type.__module__)
                file.write(f'{fn_name}: "{defined_type.__module__}.{defined_type.__name__}" = ...\n')
                continue
            rna_type: bpy.types.OperatorProperties = fn.get_rna_type()
            file.write(
                f'class {fn_name}({type(fn).__module__}.{type(fn).__name__}):  # fake class definition, real value: {fn}\n')
            file.write(indent(f'"""{fn.__doc__}"""\n', 4 * ' '))
            file.write(f'    bl_options = {fn.bl_options}\n')
            file.write(f'    idname = lambda : {repr(fn.idname())}\n')
            file.write(f'    get_rna_type: "_bpy.types.OperatorProperties" = ...\n')
            file.write(f'    """Runtime value: {rna_type}"""\n\n')
            doc = StringIO()
            args = []
            for prop in rna_type.properties:
                if prop.identifier == "rna_type":
                    continue
                doc.write(f':param {prop.identifier}: {prop.description}\n')
                if isinstance(prop, bpy.types.PointerProperty):
                    if prop.is_required:
                        args.append(f'{prop.identifier}: "{type(prop).__module__}.{type(prop).__name__}" = ...')
                    else:
                        args.append(f'{prop.identifier}: "{type(prop).__module__}.{type(prop).__name__}"')
                elif isinstance(prop, bpy.types.FloatProperty):
                    if prop.is_required:
                        args.append(f'{prop.identifier}: float = {prop.default}')
                    else:
                        args.append(f'{prop.identifier}: float')
                # elif isinstance(prop, bpy.types.FloatVectorProperty):
                #     ...
                elif isinstance(prop, bpy.types.BoolProperty):
                    if prop.is_required:
                        args.append(f'{prop.identifier}: bool = {prop.default}')
                    else:
                        args.append(f'{prop.identifier}: bool')
                # elif isinstance(prop, bpy.types.BoolVectorProperty):
                #     ...
                elif isinstance(prop, bpy.types.IntProperty):
                    if prop.is_required:
                        args.append(f'{prop.identifier}: int = {prop.default}')
                    else:
                        args.append(f'{prop.identifier}: int')
                # elif isinstance(prop, bpy.types.IntVectorProperty):
                #     ...
                elif isinstance(prop, bpy.types.CollectionProperty):
                    args.append(f'{prop.identifier}: "{type(prop).__module__}.{type(prop).__name__}" = ...')
                elif isinstance(prop, bpy.types.StringProperty):
                    if prop.is_required:
                        args.append(f'{prop.identifier}: str = {prop.default}')
                    else:
                        args.append(f'{prop.identifier}: str')
                elif isinstance(prop, bpy.types.EnumProperty):
                    enums: Iterable[bpy.types.EnumPropertyItem] = prop.enum_items
                    enums_str = ','.join(f'"{i.identifier}"' for i in enums)
                    file.write(f'    __{prop.identifier} = Literal[{enums_str}]\n')
                    if prop.is_required:
                        args.append(f'{prop.identifier}: __{prop.identifier}')
                    else:
                        args.append(f'{prop.identifier}: __{prop.identifier} = "{prop.default}"')
                    for i in enums:
                        doc.write(f'    - {i.name} ({i.identifier}) {i.description}\n')
                else:
                    logging.fatal(f'Unknown property type: {prop} from {rna_type}')
            # this hand written signature somes from bpy.ops._BPyOpsSubModOp._parse_args
            # it allows for overriding context, but there is no way to get the signature
            file.write(f'    def __init__(self, C_dict: Dict[str, Any] = ..., C_exec: str = ..., C_undo: int = ..., /')
            if args:
                file.write(', *, ')
                file.write(', '.join(args))
            file.write(') -> __OperatorResult:\n')
            file.write(indent(f'"""{bpy.ops._op_as_string(fn.idname())}\n\n'
                              f'{doc.getvalue()}"""\n\n', 8 * ' '))
        file.close()
    module2stub(
        basepath=ops_basedir, module_file_name='__init__', module_name='ops',
        module_members=inspect.getmembers(_bpy.ops),
        additional_raw_imports=set(f'from . import {mod}' for mod in ops_submodules.keys()))


def stub_mathutils():
    import mathutils
    logging.info('Generate mathutils')
    pypackage2stub(ROOT_DIR, mathutils)
    del mathutils


def stub_bpy_path():
    import _bpy_path
    logging.info('Generate _bpy_path')
    pypackage2stub(ROOT_DIR, _bpy_path)
    del _bpy_path


def stub_bgl():
    import bgl
    logging.info('Generate bgl')
    pypackage2stub(ROOT_DIR, bgl)
    del bgl


def stub_bgf():
    if bpy.app.build_options.freestyle:
        import bgf
        logging.info('Generate bgf')
        pypackage2stub(ROOT_DIR, bgf)
        del bgf
    else:
        logging.info('Skipping bgf (blender build without freestyle)')


def stub_bl_math():
    import bl_math
    logging.info('Generate bl_math')
    pypackage2stub(ROOT_DIR, bl_math)


def stub_imbuf():
    import imbuf
    logging.info('Generate imbuf')
    pypackage2stub(ROOT_DIR, imbuf)


def stub_bmesh():
    import bmesh
    logging.info('Generate bmesh')
    pypackage2stub(ROOT_DIR, bmesh)


def stub_cycles():
    if bpy.app.build_options.cycles:
        import _cycles
        logging.info('Generate _cycles')
        pypackage2stub(ROOT_DIR, _cycles)
    else:
        logging.info('Skipping cycles (blender build without cycles)')


def stub_aud():
    if bpy.app.build_options.audaspace:
        import aud
        logging.info('Generate aud')
        pypackage2stub(ROOT_DIR, aud)
    else:
        logging.info('Skipping aud (blender build without audaspace)')


def stub_manta():
    if bpy.app.build_options.fluid:
        import manta
        logging.info('Generate manta')
        pypackage2stub(ROOT_DIR, manta)
    else:
        logging.info('Skipping manta (blender build without fluid)')


def stub_gpu():
    import gpu
    logging.info('Generate gpu')
    pypackage2stub(ROOT_DIR, gpu)


def stub_idprop():
    import idprop
    logging.info('Generate idprop')
    pypackage2stub(ROOT_DIR, idprop)


if __name__ == '__main__':
    # internal modules are listed in bpy_interface.c in array: bpy_internal_modules
    stub_bpy()
    stub_mathutils()
    stub_bpy_path()
    stub_bgl()
    # stub_bgf() # todo module not found. Where is it defined?
    stub_bl_math()
    stub_imbuf()
    stub_bmesh()
    stub_manta()
    stub_aud()
    stub_cycles()
    stub_gpu()
    stub_idprop()

    logging.info('Finished')
del bpy
