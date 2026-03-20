# /// script
# requires-python = ">=3.10"
# dependencies = ["tree-sitter", "tree-sitter-cpp"]
# ///
"""Extract device adapter properties from C++ source files.

Usage:
    # auto-detect ./DeviceAdapters (+ ./SecretDeviceAdapters)
    uv run tools/parse_device_properties.py

    # specify path and output JSON file
    uv run tools/parse_device_properties.py ./DeviceAdapters -o devices.json
    uv run tools/parse_device_properties.py ./DeviceAdapters/DemoCamera
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import tree_sitter_cpp as tscpp
from tree_sitter import Language, Parser

CPP_LANG = Language(tscpp.language())
_parser = Parser(CPP_LANG)
_SCRIPT_DIR = Path(__file__).resolve().parent

# Maps C++ base class templates to device type strings
BASE_CLASS_TO_TYPE: dict[str, str] = {
    "CCameraBase": "Camera",
    "CLegacyCameraBase": "Camera",
    "CStageBase": "Stage",
    "CXYStageBase": "XYStage",
    "CShutterBase": "Shutter",
    "CStateDeviceBase": "StateDevice",
    "CGenericBase": "Generic",
    "CSerialBase": "Serial",
    "CAutoFocusBase": "AutoFocus",
    "CImageProcessorBase": "ImageProcessor",
    "CSignalIOBase": "SignalIO",
    "CMagnifierBase": "Magnifier",
    "CSLMBase": "SLM",
    "CGalvoBase": "Galvo",
    "HubBase": "Hub",
    "CVolumetricPumpBase": "VolumetricPump",
    "CPressurePumpBase": "PressurePump",
}

# Maps MM::PropertyType to JSON schema types
MM_TYPE_MAP = {
    "MM::String": "string",
    "MM::Float": "number",
    "MM::Integer": "integer",
}

# Maps convenience Create*Property methods to JSON schema types
CREATE_TYPED = {
    "CreateFloatProperty": "number",
    "CreateIntegerProperty": "integer",
    "CreateStringProperty": "string",
}

CREATE_GENERIC = {"CreateProperty", "CreatePropertyWithHandler"}

# Maps MM::DeviceType enum to our device type strings
_MM_DEVICE_TYPE_MAP = {
    "MM::CameraDevice": "Camera",
    "MM::StageDevice": "Stage",
    "MM::XYStageDevice": "XYStage",
    "MM::ShutterDevice": "Shutter",
    "MM::StateDevice": "StateDevice",
    "MM::GenericDevice": "Generic",
    "MM::SerialDevice": "Serial",
    "MM::AutoFocusDevice": "AutoFocus",
    "MM::ImageProcessorDevice": "ImageProcessor",
    "MM::SignalIODevice": "SignalIO",
    "MM::MagnifierDevice": "Magnifier",
    "MM::SLMDevice": "SLM",
    "MM::GalvoDevice": "Galvo",
    "MM::HubDevice": "Hub",
    "MM::PressurePumpDevice": "PressurePump",
    "MM::VolumetricPumpDevice": "VolumetricPump",
}

# Type alias for parsed file tuples
ParsedFile = tuple[Path, Any, bytes]


# ---------------------------------------------------------------------------
# Tree-sitter helpers
# ---------------------------------------------------------------------------


def _text(node: Any, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode()


def _walk(node: Any):
    """Depth-first walk of all nodes."""
    yield node
    for child in node.children:
        yield from _walk(child)


def _walk_toplevel(node: Any):
    """Yield top-level declarations, recursing into namespaces and preprocessor blocks."""
    for child in node.children:
        yield child
        if child.type in ("namespace_definition", "linkage_specification"):
            body = child.child_by_field_name("body")
            if body:
                yield from _walk_toplevel(body)
        elif child.type in (
            "preproc_ifdef",
            "preproc_else",
            "preproc_if",
            "declaration_list",
        ):
            yield from _walk_toplevel(child)


def parse_file(path: Path) -> tuple[Any, bytes]:
    source = path.read_bytes()
    return _parser.parse(source), source


def _get_function_name(func_node: Any, source: bytes) -> str:
    """Extract the (possibly qualified) function name from a function_definition."""
    decl = func_node.child_by_field_name("declarator")
    if decl is None:
        return ""
    for n in _walk(decl):
        if n.type == "qualified_identifier":
            return _text(n, source).replace(" ", "")
        if n.type == "function_declarator":
            name_child = n.child_by_field_name("declarator")
            if name_child and name_child.type in ("identifier", "qualified_identifier"):
                return _text(name_child, source).replace(" ", "")
    # Fallback: text up to "(" with pointer stars stripped
    text = _text(decl, source)
    paren = text.find("(")
    return text[:paren].strip().lstrip("*").strip() if paren > 0 else text.strip()


def _get_call_name(call_node: Any, source: bytes) -> str:
    """Extract the function name from a call_expression."""
    fn = call_node.child_by_field_name("function")
    if fn is None:
        return ""
    if fn.type == "identifier":
        return _text(fn, source)
    if fn.type == "field_expression":
        field = fn.child_by_field_name("field")
        return _text(field, source) if field else ""
    if fn.type == "qualified_identifier":
        return _text(fn, source).replace(" ", "")
    return _text(fn, source)


def _get_call_args(call_node: Any) -> list[Any]:
    args_node = call_node.child_by_field_name("arguments")
    return list(args_node.named_children) if args_node else []


# ---------------------------------------------------------------------------
# String / number extraction
# ---------------------------------------------------------------------------


def _extract_string_literal(node: Any, source: bytes) -> str:
    raw = _text(node, source)
    return raw[1:-1] if raw.startswith('"') and raw.endswith('"') else raw


def _extract_concatenated_string(node: Any, source: bytes) -> str:
    return "".join(
        _extract_string_literal(c, source)
        for c in node.named_children
        if c.type == "string_literal"
    )


def _parse_number(text: str, prop_type: str = "number") -> float | int | None:
    text = text.strip().rstrip("fFlLuU")
    if prop_type == "integer":
        try:
            return int(float(text))
        except ValueError:
            return None
    try:
        val = float(text)
        if val == int(val) and "." not in text and "e" not in text.lower():
            return int(val)
        return val
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Constant resolution
# ---------------------------------------------------------------------------


def parse_mm_keywords(adapters_path: Path) -> dict[str, str]:
    """Parse MM::g_Keyword_* from MMDeviceConstants.h."""
    for root in (_SCRIPT_DIR.parent, adapters_path.resolve()):
        current = root
        for _ in range(5):
            candidate = current / "MMDevice" / "MMDeviceConstants.h"
            if candidate.exists():
                return _parse_mm_keywords_file(candidate)
            current = current.parent
    return {}


def _parse_mm_keywords_file(path: Path) -> dict[str, str]:
    tree, source = parse_file(path)
    result: dict[str, str] = {}
    for node in _walk(tree.root_node):
        if node.type != "declaration":
            continue
        text = _text(node, source)
        if "g_Keyword_" not in text:
            continue
        ident = strval = None
        for child in _walk(node):
            if child.type == "identifier" and "g_Keyword_" in _text(child, source):
                ident = _text(child, source)
            if child.type == "string_literal":
                strval = _extract_string_literal(child, source)
        if ident and strval is not None:
            result[f"MM::{ident}"] = strval
            result[ident] = strval
    return result


def parse_constants(files: list[ParsedFile]) -> dict[str, str]:
    """Build identifier->string_value map from const char* and #define declarations."""
    constants: dict[str, str] = {}
    for _, tree, source in files:
        _collect_constants(tree.root_node, source, constants, namespace_prefix="")
    # Second pass: resolve constants referencing macros
    # e.g. const char* g_Foo = MACRO_NAME; where MACRO_NAME is a #define
    changed = True
    while changed:
        changed = False
        for key, val in list(constants.items()):
            if not val.startswith('"') and val in constants and constants[val] != val:
                constants[key] = constants[val]
                changed = True
    return constants


def _collect_constants(
    node: Any, source: bytes, out: dict[str, str], namespace_prefix: str
):
    """Recursively collect constants, tracking namespace context."""
    for child in node.children:
        if child.type == "declaration":
            _try_extract_const(child, source, out, namespace_prefix)
        elif child.type == "preproc_def":
            _try_extract_define(child, source, out)
        elif child.type == "namespace_definition":
            ns_name_node = child.child_by_field_name("name")
            ns_name = _text(ns_name_node, source) if ns_name_node else ""
            prefix = f"{namespace_prefix}{ns_name}::" if ns_name else namespace_prefix
            body = child.child_by_field_name("body")
            if body:
                _collect_constants(body, source, out, prefix)
        elif child.type == "namespace_alias":
            _handle_namespace_alias(child, source, out)
        elif child.type in (
            "preproc_ifdef",
            "preproc_if",
            "preproc_else",
            "declaration_list",
            "linkage_specification",
        ):
            body = child.child_by_field_name("body")
            if body:
                _collect_constants(body, source, out, namespace_prefix)
            else:
                _collect_constants(child, source, out, namespace_prefix)


def _handle_namespace_alias(node: Any, source: bytes, out: dict[str, str]):
    """Handle 'namespace Alias = Original;' by duplicating constants."""
    name_node = node.child_by_field_name("name")
    original = None
    for child in node.children:
        if child.type in ("identifier", "qualified_identifier"):
            original = _text(child, source).strip()
    alias = _text(name_node, source).strip() if name_node else None
    if not alias or not original or alias == original:
        return
    orig_prefix = f"{original}::"
    alias_prefix = f"{alias}::"
    for key, val in list(out.items()):
        if key.startswith(orig_prefix):
            out[alias_prefix + key[len(orig_prefix) :]] = val


def _try_extract_define(node: Any, source: bytes, out: dict[str, str]):
    """Extract '#define FOO "bar"' patterns."""
    name_node = node.child_by_field_name("name")
    value_node = node.child_by_field_name("value")
    if name_node is None or value_node is None:
        return
    name = _text(name_node, source).strip()
    val_text = _text(value_node, source).strip()
    if val_text.startswith('"') and val_text.endswith('"'):
        out[name] = val_text[1:-1]
    elif val_text in out:
        out[name] = out[val_text]


def _try_extract_const(
    decl_node: Any, source: bytes, out: dict[str, str], namespace_prefix: str = ""
):
    """Extract 'const char* name = "value"' and 'constexpr char name[] = "value"'."""
    if "char" not in _text(decl_node, source):
        return
    for node in _walk(decl_node):
        if node.type != "init_declarator":
            continue
        decl = node.child_by_field_name("declarator")
        val_node = node.child_by_field_name("value")
        if decl is None or val_node is None:
            continue
        # Find identifier inside any declarator shape
        ident = next(
            (_text(n, source) for n in _walk(decl) if n.type == "identifier"), None
        )
        # Find string value (literal, concatenated, or macro identifier)
        strval = None
        if val_node.type == "string_literal":
            strval = _extract_string_literal(val_node, source)
        elif val_node.type == "concatenated_string":
            strval = _extract_concatenated_string(val_node, source)
        elif val_node.type == "identifier":
            strval = _text(val_node, source)
        if ident and strval is not None:
            out[ident] = strval
            if namespace_prefix:
                out[f"{namespace_prefix}{ident}"] = strval


# ---------------------------------------------------------------------------
# Class detection
# ---------------------------------------------------------------------------


def find_device_classes(files: list[ParsedFile]) -> list[dict]:
    """Find all classes extending known device base classes."""
    classes = []
    for path, tree, source in files:
        for node in _walk_toplevel(tree.root_node):
            if node.type != "class_specifier":
                continue
            name_node = node.child_by_field_name("name")
            if name_node is None:
                continue
            class_name = _text(name_node, source)
            for child in node.children:
                if child.type != "base_class_clause":
                    continue
                for base_child in _walk(child):
                    if base_child.type == "template_type":
                        type_node = base_child.child_by_field_name("name")
                        base_name = _text(type_node, source) if type_node else None
                    elif base_child.type == "type_identifier":
                        base_name = _text(base_child, source)
                    else:
                        continue
                    if base_name in BASE_CLASS_TO_TYPE:
                        classes.append(
                            {
                                "class_name": class_name,
                                "base_class": base_name,
                                "device_type": BASE_CLASS_TO_TYPE[base_name],
                                "file": path,
                            }
                        )
    return classes


# ---------------------------------------------------------------------------
# Device name resolution
# ---------------------------------------------------------------------------


def _extract_getname_arg(
    func_node: Any, source: bytes, constants: dict[str, str]
) -> str:
    """Extract the name string from a GetName method body."""
    body = func_node.child_by_field_name("body")
    if body is None:
        return "<unresolved:GetName:nobody>"
    for node in _walk(body):
        if node.type != "call_expression":
            continue
        fn = _get_call_name(node, source)
        if fn in ("CDeviceUtils::CopyLimitedString", "strcpy"):
            args = _get_call_args(node)
            if len(args) >= 2:
                text = _text(args[1], source).strip()
                if text.startswith('"') and text.endswith('"'):
                    return text[1:-1]
                return constants.get(
                    text, constants.get(text.lstrip(":"), f"<unresolved:{text}>")
                )
    return "<unresolved:GetName>"


def resolve_device_name(
    class_name: str, files: list[ParsedFile], constants: dict[str, str]
) -> str:
    """Find the string returned by ClassName::GetName."""
    for _, tree, source in files:
        for node in _walk_toplevel(tree.root_node):
            if node.type == "function_definition":
                name = _get_function_name(node, source)
                if name == f"{class_name}::GetName":
                    return _extract_getname_arg(node, source, constants)
            if node.type == "class_specifier":
                cname = node.child_by_field_name("name")
                if cname and _text(cname, source) == class_name:
                    # Check for inline GetName in class body
                    body = node.child_by_field_name("body")
                    if body is None:
                        continue
                    for fn in _walk(body):
                        if fn.type != "function_definition":
                            continue
                        decl = fn.child_by_field_name("declarator")
                        if decl and "GetName" in _text(decl, source):
                            return _extract_getname_arg(fn, source, constants)
    return f"<unresolved:{class_name}::GetName>"


# ---------------------------------------------------------------------------
# Property extraction
# ---------------------------------------------------------------------------


def _get_inline_method_name(func_node: Any, source: bytes) -> str | None:
    """Get the simple method name from an inline function definition."""
    decl = func_node.child_by_field_name("declarator")
    if decl is None:
        return None
    for n in _walk(decl):
        if n.type == "function_declarator":
            name_child = n.child_by_field_name("declarator")
            if name_child and name_child.type in ("identifier", "field_identifier"):
                return _text(name_child, source)
            break
    return None


def _classify_method(
    method_name: str,
    body: Any,
    source: bytes,
    class_name: str,
    target_bodies: list,
    method_bodies: dict,
):
    """Route a method body to target_bodies or method_bodies."""
    if method_name == class_name:
        target_bodies.append((body, source, True))
    elif method_name == "Initialize":
        target_bodies.append((body, source, False))
    else:
        method_bodies[method_name] = (body, source)


def extract_properties(
    class_name: str, files: list[ParsedFile], constants: dict[str, str]
) -> dict[str, dict]:
    """Extract all properties from constructor, Initialize, and their helpers."""
    properties: dict[str, dict] = {}
    method_bodies: dict[str, tuple] = {}
    target_bodies: list[tuple] = []

    for _, tree, source in files:
        for node in _walk_toplevel(tree.root_node):
            # Out-of-line definitions: ClassName::Method()
            if node.type == "function_definition":
                fname = _get_function_name(node, source)
                if "::" in fname:
                    cls, method = fname.rsplit("::", 1)
                    if cls == class_name:
                        body = node.child_by_field_name("body")
                        if body:
                            _classify_method(
                                method,
                                body,
                                source,
                                class_name,
                                target_bodies,
                                method_bodies,
                            )

            # Inline definitions inside class body
            if node.type == "class_specifier":
                cname = node.child_by_field_name("name")
                if cname and _text(cname, source) == class_name:
                    class_body = node.child_by_field_name("body")
                    if class_body:
                        for fn in _walk(class_body):
                            if fn.type != "function_definition":
                                continue
                            method_name = _get_inline_method_name(fn, source)
                            fn_body = fn.child_by_field_name("body")
                            if method_name and fn_body:
                                _classify_method(
                                    method_name,
                                    fn_body,
                                    source,
                                    class_name,
                                    target_bodies,
                                    method_bodies,
                                )

    for body, source, _is_ctor in target_bodies:
        local_vars: dict[str, str] = {}
        _extract_from_body(body, source, constants, local_vars, properties)

        # Follow helper method calls one level deep
        for fn_node in _walk(body):
            if fn_node.type == "call_expression":
                fn_name = _get_call_name(fn_node, source)
                if fn_name in method_bodies:
                    helper_body, helper_source = method_bodies[fn_name]
                    _extract_from_body(
                        helper_body, helper_source, constants, {}, properties
                    )

    return properties


# ---------------------------------------------------------------------------
# Statement-level property extraction with variable tracking
# ---------------------------------------------------------------------------


def _try_resolve_node_value(
    node: Any, source: bytes, constants: dict[str, str]
) -> str | None:
    """Try to resolve a node to a string value."""
    if node.type == "string_literal":
        return _extract_string_literal(node, source)
    if node.type == "concatenated_string":
        return _extract_concatenated_string(node, source)
    if node.type == "identifier":
        return constants.get(_text(node, source))
    if node.type == "qualified_identifier":
        return constants.get(_text(node, source).replace(" ", ""))
    return None


def _extract_from_body(
    body_node: Any,
    source: bytes,
    constants: dict[str, str],
    local_vars: dict[str, str],
    properties: dict[str, dict],
):
    """Walk a function body extracting property creation and configuration calls."""
    for stmt in body_node.named_children:
        _process_statement(stmt, source, constants, local_vars, properties, body_node)


def _process_statement(
    node: Any,
    source: bytes,
    constants: dict[str, str],
    local_vars: dict[str, str],
    properties: dict[str, dict],
    body_node: Any,
):
    """Process a single statement, tracking variable state and extracting properties."""

    def handle_call(call_node: Any):
        _process_call(call_node, source, constants, local_vars, properties, body_node)

    # Track variable declarations: std::string propName = "Foo";
    if node.type == "declaration":
        for child in node.named_children:
            if child.type == "init_declarator":
                decl = child.child_by_field_name("declarator")
                val_node = child.child_by_field_name("value")
                if decl and val_node:
                    ident = next(
                        (
                            _text(n, source)
                            for n in _walk(decl)
                            if n.type == "identifier"
                        ),
                        None,
                    )
                    if ident:
                        val = _try_resolve_node_value(val_node, source, constants)
                        if val is not None:
                            local_vars[ident] = val
        for sub in _walk(node):
            if sub.type == "call_expression":
                handle_call(sub)
        return

    # Track variable reassignments: propName = "Bar";
    if node.type == "expression_statement":
        for sub in node.named_children:
            if sub.type == "assignment_expression":
                left = sub.child_by_field_name("left")
                right = sub.child_by_field_name("right")
                if left and right:
                    val = _try_resolve_node_value(right, source, constants)
                    if val is not None:
                        local_vars[_text(left, source).strip()] = val
                    if right.type == "call_expression":
                        handle_call(right)
            elif sub.type == "call_expression":
                handle_call(sub)
        return

    # Recurse into compound/control-flow statements
    if node.type == "compound_statement":
        for child in node.named_children:
            _process_statement(
                child, source, constants, local_vars, properties, body_node
            )
        return

    if node.type in ("if_statement", "for_statement", "while_statement"):
        for child in node.named_children:
            if child.type == "compound_statement":
                _process_statement(
                    child, source, constants, local_vars, properties, body_node
                )
        return

    # Walk anything else looking for call expressions
    for sub in _walk(node):
        if sub.type == "call_expression":
            handle_call(sub)


def _process_call(
    node: Any,
    source: bytes,
    constants: dict[str, str],
    local_vars: dict[str, str],
    properties: dict[str, dict],
    body_node: Any,
):
    """Process a single call_expression node."""
    fn_name = _get_call_name(node, source)
    args = _get_call_args(node)

    if fn_name in CREATE_TYPED:
        _handle_create_property(
            args,
            source,
            constants,
            local_vars,
            properties,
            prop_type=CREATE_TYPED[fn_name],
            type_arg_idx=None,
            readonly_arg_idx=2,
            preinit_arg_idx=4,
        )
    elif fn_name in CREATE_GENERIC:
        _handle_create_property(
            args,
            source,
            constants,
            local_vars,
            properties,
            prop_type=None,
            type_arg_idx=2,
            readonly_arg_idx=3,
            preinit_arg_idx=5,
        )
    elif fn_name == "SetPropertyLimits":
        _handle_set_limits(args, source, constants, local_vars, properties)
    elif fn_name == "AddAllowedValue":
        _handle_add_allowed(args, source, constants, local_vars, properties)
    elif fn_name == "SetAllowedValues":
        _handle_set_allowed_values(
            args, source, constants, local_vars, properties, body_node
        )


# ---------------------------------------------------------------------------
# Property creation/configuration handlers
# ---------------------------------------------------------------------------


def _handle_create_property(
    args: list,
    source: bytes,
    constants: dict[str, str],
    local_vars: dict[str, str],
    properties: dict[str, dict],
    *,
    prop_type: str | None,
    type_arg_idx: int | None,
    readonly_arg_idx: int,
    preinit_arg_idx: int,
):
    """Unified handler for all Create*Property variants."""
    min_args = readonly_arg_idx + 1
    if len(args) < min_args:
        return
    if prop_type is None and type_arg_idx is not None:
        prop_type = MM_TYPE_MAP.get(_text(args[type_arg_idx], source).strip(), "string")

    prop_name = _resolve_arg(args[0], source, constants, local_vars)
    default_val = _resolve_default(args[1], source, constants, local_vars, prop_type)
    read_only = _resolve_bool(args[readonly_arg_idx], source)
    pre_init = (
        _resolve_bool(args[preinit_arg_idx], source)
        if len(args) > preinit_arg_idx
        else False
    )

    prop: dict = {"type": prop_type}
    if read_only:
        prop["readOnly"] = True
    if default_val is not None:
        prop["default"] = default_val
    if pre_init:
        prop["preInit"] = True
    properties[prop_name] = prop


def _handle_set_limits(
    args: list,
    source: bytes,
    constants: dict[str, str],
    local_vars: dict[str, str],
    properties: dict[str, dict],
):
    if len(args) < 3:
        return
    prop_name = _resolve_arg(args[0], source, constants, local_vars)
    if prop_name not in properties:
        return
    for key, idx in (("minimum", 1), ("maximum", 2)):
        val = _resolve_number(args[idx], source, constants, local_vars)
        properties[prop_name][key] = (
            val if val is not None else f"<unresolved:{_text(args[idx], source)}>"
        )


def _handle_add_allowed(
    args: list,
    source: bytes,
    constants: dict[str, str],
    local_vars: dict[str, str],
    properties: dict[str, dict],
):
    if len(args) < 2:
        return
    prop_name = _resolve_arg(args[0], source, constants, local_vars)
    value = _resolve_arg(args[1], source, constants, local_vars)
    if prop_name in properties:
        enum = properties[prop_name].setdefault("enum", [])
        if value not in enum:
            enum.append(value)


def _handle_set_allowed_values(
    args: list,
    source: bytes,
    constants: dict[str, str],
    local_vars: dict[str, str],
    properties: dict[str, dict],
    body_node: Any,
):
    """Handle SetAllowedValues(name, vector_var)."""
    if len(args) < 2:
        return
    prop_name = _resolve_arg(args[0], source, constants, local_vars)
    if prop_name not in properties:
        return
    vec_var = _text(args[1], source).strip()
    values = _trace_vector_values(vec_var, body_node, source, constants, local_vars)
    if values:
        properties[prop_name]["enum"] = values


def _trace_vector_values(
    vec_name: str,
    body_node: Any,
    source: bytes,
    constants: dict[str, str],
    local_vars: dict[str, str],
) -> list[str]:
    """Trace vector values from push_back() calls or brace initialization."""
    values: list[str] = []
    for node in _walk(body_node):
        # vec.push_back(val)
        if node.type == "call_expression":
            fn_node = node.child_by_field_name("function")
            if fn_node and fn_node.type == "field_expression":
                obj = fn_node.child_by_field_name("argument")
                field = fn_node.child_by_field_name("field")
                if (
                    obj
                    and field
                    and _text(obj, source).strip() == vec_name
                    and _text(field, source) == "push_back"
                ):
                    call_args = _get_call_args(node)
                    if call_args:
                        val = _resolve_arg(call_args[0], source, constants, local_vars)
                        if val not in values:
                            values.append(val)
        # vector<string> vec = {"a", "b"}
        if node.type == "declaration":
            for child in node.named_children:
                if child.type != "init_declarator":
                    continue
                decl = child.child_by_field_name("declarator")
                if decl is None:
                    continue
                ident = next(
                    (_text(n, source) for n in _walk(decl) if n.type == "identifier"),
                    None,
                )
                if ident != vec_name:
                    continue
                val_node = child.child_by_field_name("value")
                if val_node and val_node.type == "initializer_list":
                    for item in val_node.named_children:
                        values.append(_resolve_arg(item, source, constants, local_vars))
    return values


# ---------------------------------------------------------------------------
# Boolean detection and enum coercion
# ---------------------------------------------------------------------------


def detect_booleans(properties: dict[str, dict]):
    for prop in properties.values():
        if prop.get("type") == "integer" and prop.get("enum") == ["0", "1"]:
            prop["type"] = "boolean"
            del prop["enum"]
            if "default" in prop:
                prop["default"] = bool(prop["default"])


def coerce_enum_types(properties: dict[str, dict]):
    """Convert enum string values to numbers for integer/number typed properties."""
    for prop in properties.values():
        if "enum" not in prop:
            continue
        ptype = prop.get("type")
        if ptype == "integer":
            try:
                prop["enum"] = sorted(int(v) for v in prop["enum"])
            except (ValueError, TypeError):
                pass
        elif ptype == "number":
            try:
                prop["enum"] = sorted(float(v) for v in prop["enum"])
            except (ValueError, TypeError):
                pass
        elif ptype == "string":
            prop["enum"] = sorted(prop["enum"])


# ---------------------------------------------------------------------------
# Argument resolution
# ---------------------------------------------------------------------------


def _resolve_arg(
    node: Any, source: bytes, constants: dict[str, str], local_vars: dict[str, str]
) -> str:
    """Resolve a call argument to a string value."""
    text = _text(node, source).strip()

    if node.type == "string_literal":
        return _extract_string_literal(node, source)
    if node.type == "concatenated_string":
        return _extract_concatenated_string(node, source)

    if node.type == "qualified_identifier":
        qname = text.replace(" ", "")
        if qname in constants:
            return constants[qname]
        # Try resolving with different namespace prefix (handles aliases)
        if "::" in qname:
            suffix = qname.split("::", 1)[1]
            for key, val in constants.items():
                if key.endswith(f"::{suffix}") and key != qname:
                    return val

    if node.type == "identifier":
        if text in constants:
            return constants[text]
        if text in local_vars:
            return local_vars[text]

    # foo.c_str() -> look up foo
    if node.type == "call_expression":
        fn = node.child_by_field_name("function")
        if fn and fn.type == "field_expression":
            field = fn.child_by_field_name("field")
            obj = fn.child_by_field_name("argument")
            if field and _text(field, source) == "c_str" and obj:
                obj_name = _text(obj, source).strip()
                return (
                    local_vars.get(obj_name)
                    or constants.get(obj_name)
                    or f"<unresolved:{obj_name}>"
                )

    # ::g_Foo (global scope prefix)
    if text.startswith("::"):
        bare = text[2:]
        if bare in constants:
            return constants[bare]

    return f"<unresolved:{text}>"


def _resolve_default(
    node: Any,
    source: bytes,
    constants: dict[str, str],
    local_vars: dict[str, str],
    prop_type: str,
) -> Any:
    """Resolve a default value, returning typed value or None."""
    text = _text(node, source).strip()

    if node.type == "number_literal":
        return _parse_number(text, prop_type)
    if node.type == "string_literal":
        return _extract_string_literal(node, source)
    if node.type == "unary_expression":
        return _parse_number(text, prop_type)

    if node.type == "identifier":
        if text in constants:
            val = constants[text]
            if prop_type in ("number", "integer"):
                try:
                    return _parse_number(val, prop_type)
                except (ValueError, TypeError):
                    return val
            return val
        if text in local_vars:
            return local_vars[text]
        return None  # Member variable — omit

    if node.type == "qualified_identifier":
        qname = text.replace(" ", "")
        return constants.get(qname)

    return None


def _resolve_bool(node: Any, source: bytes) -> bool:
    return node is not None and _text(node, source).strip() == "true"


def _resolve_number(
    node: Any, source: bytes, constants: dict[str, str], local_vars: dict[str, str]
) -> float | int | None:
    text = _text(node, source).strip()
    if node.type in ("number_literal", "unary_expression"):
        return _parse_number(text)
    if node.type == "identifier" and text in constants:
        return _parse_number(constants[text])
    return _parse_number(text)


# ---------------------------------------------------------------------------
# CreateDevice / RegisterDevice name resolution
# ---------------------------------------------------------------------------


def parse_create_device(
    files: list[ParsedFile], constants: dict[str, str]
) -> dict[str, str]:
    """Parse CreateDevice() to map class names to device name constants."""
    class_to_name: dict[str, str] = {}
    for _, tree, source in files:
        for node in _walk_toplevel(tree.root_node):
            if node.type != "function_definition":
                continue
            fname = _get_function_name(node, source)
            if fname not in ("CreateDevice", "::CreateDevice"):
                continue
            body = node.child_by_field_name("body")
            if body is None:
                continue
            _extract_create_device_mappings(body, source, constants, class_to_name)
    return class_to_name


def _extract_create_device_mappings(
    node: Any, source: bytes, constants: dict[str, str], out: dict[str, str]
):
    """Walk CreateDevice body extracting strcmp->new Class mappings."""
    last_strcmp_name: str | None = None
    for child in _walk(node):
        if child.type == "call_expression":
            fn = _get_call_name(child, source)
            if fn == "strcmp":
                args = _get_call_args(child)
                if len(args) >= 2:
                    const_text = _text(args[1], source).strip()
                    last_strcmp_name = constants.get(
                        const_text, constants.get(const_text.lstrip(":"), const_text)
                    )

        if child.type == "new_expression" and last_strcmp_name:
            type_node = child.child_by_field_name("type")
            if type_node:
                cname = _text(type_node, source).strip()
                if cname and "<" not in cname:
                    out[cname] = last_strcmp_name

        if child.type == "binary_expression":
            op = child.child_by_field_name("operator")
            if op and _text(op, source).strip() == "==":
                right = child.child_by_field_name("right")
                if right:
                    right_text = _text(right, source).strip()
                    if right_text.startswith('"'):
                        last_strcmp_name = right_text.strip('"')
                    elif right_text in constants:
                        last_strcmp_name = constants[right_text]


def parse_register_device_calls(
    files: list[ParsedFile], constants: dict[str, str]
) -> list[dict]:
    """Parse RegisterDevice(name, type, description) calls."""
    results = []
    for _, tree, source in files:
        for node in _walk(tree.root_node):
            if node.type != "call_expression":
                continue
            if _get_call_name(node, source) != "RegisterDevice":
                continue
            args = _get_call_args(node)
            if len(args) < 3:
                continue

            name_text = _text(args[0], source).strip()
            if name_text.startswith('"'):
                name = name_text.strip('"')
            else:
                name = constants.get(name_text, f"<unresolved:{name_text}>")

            type_text = _text(args[1], source).strip()
            desc_text = _text(args[2], source).strip()

            results.append(
                {
                    "name": name,
                    "device_type": _MM_DEVICE_TYPE_MAP.get(type_text, type_text),
                    "description": desc_text.strip('"')
                    if desc_text.startswith('"')
                    else "",
                }
            )
    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def to_device_schema(
    library: str, device_name: str, device_type: str, properties: dict[str, dict]
) -> dict:
    desc_prop = properties.get("Description", {})
    description = str(desc_prop["default"]) if "default" in desc_prop else ""
    return {
        "library": library,
        "device": device_name,
        "description": description,
        "device_type": device_type,
        "type": "object",
        "properties": dict(sorted(properties.items())),
    }


# ---------------------------------------------------------------------------
# Per-directory processing
# ---------------------------------------------------------------------------


def process_adapter_dir(adapter_dir: Path, mm_keywords: dict[str, str]) -> list[dict]:
    """Process a single adapter directory, returning device schemas."""
    source_files = list(adapter_dir.glob("*.cpp")) + list(adapter_dir.glob("*.h"))
    if not source_files:
        return []

    parsed: list[ParsedFile] = []
    for f in source_files:
        try:
            tree, source = parse_file(f)
            parsed.append((f, tree, source))
        except Exception as e:
            print(f"Warning: could not parse {f}: {e}", file=sys.stderr)

    constants = parse_constants(parsed)
    constants.update(mm_keywords)
    classes = find_device_classes(parsed)
    if not classes:
        return []

    library_name = adapter_dir.name
    registered = parse_register_device_calls(parsed, constants)
    create_device_map = parse_create_device(parsed, constants)
    results = []

    for cls in classes:
        class_name = cls["class_name"]
        device_type = cls["device_type"]

        # Resolve device name with cascading fallbacks
        device_name = resolve_device_name(class_name, parsed, constants)
        if "<unresolved" in device_name and class_name in create_device_map:
            device_name = create_device_map[class_name]
        if "<unresolved" in device_name:
            for reg in registered:
                if (
                    reg["device_type"] == device_type
                    and "<unresolved" not in reg["name"]
                ):
                    device_name = reg["name"]
                    registered.remove(reg)
                    break

        properties = extract_properties(class_name, parsed, constants)

        # Use "Name" property default as final fallback for device name
        if "<unresolved" in device_name:
            name_default = properties.get("Name", {}).get("default")
            if name_default and "<unresolved" not in str(name_default):
                device_name = str(name_default)

        # Fill description from RegisterDevice if missing
        if not properties.get("Description", {}).get("default"):
            for reg in registered:
                if reg["name"] == device_name and reg.get("description"):
                    properties["Description"] = {
                        "type": "string",
                        "readOnly": True,
                        "default": reg["description"],
                    }
                    break

        detect_booleans(properties)
        coerce_enum_types(properties)
        results.append(
            to_device_schema(library_name, device_name, device_type, properties)
        )

    return results


# ---------------------------------------------------------------------------
# Stats reporting
# ---------------------------------------------------------------------------


def _print_stats(devices: list[dict], elapsed: float) -> None:
    from collections import Counter

    total = len(devices)
    libraries = len({d["library"] for d in devices})
    resolved_names = sum(1 for d in devices if "<unresolved" not in d["device"])
    with_props = sum(1 for d in devices if d["properties"])
    total_props = sum(len(d["properties"]) for d in devices)

    unresolved_props = 0
    total_values = 0
    unresolved_values = 0
    for d in devices:
        for pname, pval in d["properties"].items():
            if "<unresolved" in pname:
                unresolved_props += 1
            for v in pval.values():
                total_values += 1
                if isinstance(v, str) and "<unresolved" in v:
                    unresolved_values += 1

    w = 30
    print("\n--- Extraction Stats ---", file=sys.stderr)
    print(f"{'Time:':{w}} {elapsed:.1f}s", file=sys.stderr)
    print(f"{'Libraries:':{w}} {libraries}", file=sys.stderr)
    print(f"{'Devices:':{w}} {total}", file=sys.stderr)
    print(
        f"{'Resolved device names:':{w}} {resolved_names}/{total} ({100 * resolved_names / total:.0f}%)",
        file=sys.stderr,
    )
    print(
        f"{'Devices with properties:':{w}} {with_props}/{total} ({100 * with_props / total:.0f}%)",
        file=sys.stderr,
    )
    print(
        f"{'Unresolved property names:':{w}} {unresolved_props}/{total_props} ({100 * unresolved_props / total_props:.0f}%)",
        file=sys.stderr,
    )
    print(
        f"{'Unresolved property values:':{w}} {unresolved_values}/{total_values} ({100 * unresolved_values / total_values:.0f}%)",
        file=sys.stderr,
    )

    print("\nBy device type:", file=sys.stderr)
    for dtype, count in Counter(d["device_type"] for d in devices).most_common():
        print(f"  {dtype:20s} {count}", file=sys.stderr)

    bad = [d for d in devices if "<unresolved" in d["device"]]
    if bad:
        print(f"\nUnresolved device names ({len(bad)}):", file=sys.stderr)
        for d in bad:
            print(f"  {d['library']:30s} {d['device']}", file=sys.stderr)

    empty = [d for d in devices if not d["properties"]]
    if empty:
        print(f"\nDevices with 0 properties ({len(empty)}):", file=sys.stderr)
        for d in empty:
            print(f"  {d['library']:30s} {d['device']}", file=sys.stderr)

    print(file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Extract device adapter properties from C++ source files."
    )
    parser.add_argument(
        "path",
        type=Path,
        nargs="*",
        help="Paths to DeviceAdapters directories (default: ./DeviceAdapters, "
        "and ./SecretDeviceAdapters if it exists)",
    )
    parser.add_argument("-o", "--output", type=Path, help="Output JSON file")
    parser.add_argument("--stats", action="store_true", help="Print extraction stats")
    args = parser.parse_args()

    t0 = time.monotonic()

    if args.path:
        adapter_roots = [p.resolve() for p in args.path]
    else:
        adapter_roots = [
            Path.cwd() / name
            for name in ("DeviceAdapters", "SecretDeviceAdapters")
            if (Path.cwd() / name).is_dir()
        ]
        if not adapter_roots:
            print("Error: no DeviceAdapters directory found in cwd", file=sys.stderr)
            sys.exit(1)

    for p in adapter_roots:
        if not p.is_dir():
            print(f"Error: {p} is not a directory", file=sys.stderr)
            sys.exit(1)

    mm_keywords = parse_mm_keywords(adapter_roots[0])
    if mm_keywords:
        print(
            f"Parsed {len(mm_keywords) // 2} MM keywords from MMDeviceConstants.h",
            file=sys.stderr,
        )
    else:
        print("Warning: could not find MMDeviceConstants.h", file=sys.stderr)

    all_devices: list[dict] = []
    for adapters_path in adapter_roots:
        has_cpp = any(adapters_path.glob("*.cpp"))
        dirs = (
            [adapters_path]
            if has_cpp
            else sorted(p for p in adapters_path.iterdir() if p.is_dir())
        )
        for subdir in dirs:
            try:
                all_devices.extend(process_adapter_dir(subdir, mm_keywords))
            except Exception as e:
                print(f"Warning: error processing {subdir.name}: {e}", file=sys.stderr)

    print(
        f"Found {len(all_devices)} devices across {', '.join(p.name for p in adapter_roots)}",
        file=sys.stderr,
    )

    if args.stats:
        _print_stats(all_devices, time.monotonic() - t0)

    output = json.dumps(all_devices, indent=2)
    if args.output:
        args.output.write_text(output)
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
