"""
Tool schema sanitization for Antigravity API.

Ported from opencode-antigravity-auth/src/plugin/request-helpers.ts
"""

from typing import Any

try:
    from .constants import (
        EMPTY_SCHEMA_PLACEHOLDER_NAME,
        EMPTY_SCHEMA_PLACEHOLDER_DESCRIPTION,
    )
except ImportError:  # pragma: no cover
    from constants import (  # type: ignore
        EMPTY_SCHEMA_PLACEHOLDER_NAME,
        EMPTY_SCHEMA_PLACEHOLDER_DESCRIPTION,
    )

# Unsupported constraint keywords that should be moved to description hints
UNSUPPORTED_CONSTRAINTS = [
    "minLength", "maxLength", "exclusiveMinimum", "exclusiveMaximum",
    "pattern", "minItems", "maxItems", "format",
    "default", "examples",
]

# Keywords that should be removed after hint extraction
UNSUPPORTED_KEYWORDS = [
    *UNSUPPORTED_CONSTRAINTS,
    "$schema", "$defs", "definitions", "const", "$ref", "additionalProperties",
    "propertyNames", "title", "$id", "$comment",
]


def append_description_hint(schema: dict[str, Any], hint: str) -> dict[str, Any]:
    """Appends a hint to a schema's description field."""
    if not isinstance(schema, dict):
        return schema
    existing = schema.get("description", "")
    new_description = f"{existing} ({hint})" if existing else hint
    return {**schema, "description": new_description}


def convert_refs_to_hints(schema: Any) -> Any:
    """Convert $ref to description hints."""
    if not isinstance(schema, dict):
        return schema
    
    if isinstance(schema, list):
        return [convert_refs_to_hints(item) for item in schema]
    
    # If this object has $ref, replace it with a hint
    if "$ref" in schema:
        ref_val = schema["$ref"]
        def_name = ref_val.split("/")[-1] if "/" in ref_val else ref_val
        hint = f"See: {def_name}"
        existing_desc = schema.get("description", "")
        new_description = f"{existing_desc} ({hint})" if existing_desc else hint
        return {"type": "object", "description": new_description}
    
    # Recursively process all properties
    result = {}
    for key, value in schema.items():
        result[key] = convert_refs_to_hints(value)
    return result


def convert_const_to_enum(schema: Any) -> Any:
    """Convert const to enum."""
    if not isinstance(schema, dict):
        return schema
    
    if isinstance(schema, list):
        return [convert_const_to_enum(item) for item in schema]
    
    result = {}
    for key, value in schema.items():
        if key == "const" and "enum" not in schema:
            result["enum"] = [value]
        else:
            result[key] = convert_const_to_enum(value)
    return result


def add_enum_hints(schema: Any) -> Any:
    """Add enum hints to description."""
    if not isinstance(schema, dict):
        return schema
    
    if isinstance(schema, list):
        return [add_enum_hints(item) for item in schema]
    
    result = {**schema}
    
    # Add enum hint if enum has 2-10 items
    if "enum" in result and isinstance(result["enum"], list):
        if 1 < len(result["enum"]) <= 10:
            vals = ", ".join(str(v) for v in result["enum"])
            result = append_description_hint(result, f"Allowed: {vals}")
    
    # Recursively process nested objects
    for key, value in result.items():
        if key != "enum" and isinstance(value, (dict, list)):
            result[key] = add_enum_hints(value)
    
    return result


def merge_all_of(schema: Any) -> Any:
    """Merge allOf schemas into a single object."""
    if not isinstance(schema, dict):
        return schema
    
    if isinstance(schema, list):
        return [merge_all_of(item) for item in schema]
    
    result = {**schema}
    
    # If this object has allOf, merge its contents
    if "allOf" in result and isinstance(result["allOf"], list):
        merged: dict[str, Any] = {}
        merged_required: list[str] = []
        
        for item in result["allOf"]:
            if not isinstance(item, dict):
                continue
            
            # Merge properties
            if "properties" in item and isinstance(item["properties"], dict):
                merged.setdefault("properties", {}).update(item["properties"])
            
            # Merge required arrays
            if "required" in item and isinstance(item["required"], list):
                for req in item["required"]:
                    if req not in merged_required:
                        merged_required.append(req)
            
            # Copy other fields
            for key, value in item.items():
                if key not in ("properties", "required") and key not in merged:
                    merged[key] = value
        
        # Apply merged content
        if merged.get("properties"):
            result.setdefault("properties", {}).update(merged["properties"])
        if merged_required:
            existing_required = result.get("required", [])
            result["required"] = list(set(existing_required + merged_required))
        
        # Copy other merged fields
        for key, value in merged.items():
            if key not in ("properties", "required") and key not in result:
                result[key] = value
        
        del result["allOf"]
    
    # Recursively process nested objects
    for key, value in list(result.items()):
        if isinstance(value, (dict, list)):
            result[key] = merge_all_of(value)
    
    return result


def try_merge_enum_from_union(options: list[Any]) -> list[str] | None:
    """Check if anyOf/oneOf represents enum choices."""
    if not isinstance(options, list) or len(options) == 0:
        return None
    
    enum_values: list[str] = []
    
    for option in options:
        if not isinstance(option, dict):
            return None
        
        # Check for const value
        if "const" in option:
            enum_values.append(str(option["const"]))
            continue
        
        # Check for single-value enum
        if "enum" in option and isinstance(option["enum"], list) and len(option["enum"]) == 1:
            enum_values.append(str(option["enum"][0]))
            continue
        
        # Check for multi-value enum
        if "enum" in option and isinstance(option["enum"], list) and len(option["enum"]) > 0:
            for val in option["enum"]:
                enum_values.append(str(val))
            continue
        
        # If option has complex structure, it's not a simple enum
        if any(key in option for key in ("properties", "items", "anyOf", "oneOf", "allOf")):
            return None
        
        # If option has only type (no const/enum), it's not an enum pattern
        if "type" in option and "const" not in option and "enum" not in option:
            return None
    
    return enum_values if enum_values else None


def flatten_any_of_one_of(schema: Any) -> Any:
    """Flatten anyOf/oneOf to the best option with type hints."""
    if not isinstance(schema, dict):
        return schema
    
    if isinstance(schema, list):
        return [flatten_any_of_one_of(item) for item in schema]
    
    result = {**schema}
    
    for union_key in ("anyOf", "oneOf"):
        if union_key in result and isinstance(result[union_key], list) and len(result[union_key]) > 0:
            options = result[union_key]
            parent_desc = result.get("description", "")
            
            # Check if this is an enum pattern
            merged_enum = try_merge_enum_from_union(options)
            if merged_enum is not None:
                new_result = {k: v for k, v in result.items() if k != union_key}
                new_result["type"] = "string"
                new_result["enum"] = merged_enum
                if parent_desc:
                    new_result["description"] = parent_desc
                result = new_result
                continue
            
            # Not an enum - select first option
            selected = flatten_any_of_one_of(options[0]) or {"type": "string"}
            
            # Preserve parent description
            if parent_desc:
                child_desc = selected.get("description", "")
                if child_desc and child_desc != parent_desc:
                    selected = {**selected, "description": f"{parent_desc} ({child_desc})"}
                elif not child_desc:
                    selected = {**selected, "description": parent_desc}
            
            # Replace result
            new_result = {k: v for k, v in result.items() if k not in (union_key, "description")}
            result = {**new_result, **selected}
    
    # Recursively process nested objects
    for key, value in list(result.items()):
        if isinstance(value, (dict, list)):
            result[key] = flatten_any_of_one_of(value)
    
    return result


def remove_unsupported_keywords(schema: Any, inside_properties: bool = False) -> Any:
    """Remove unsupported keywords after hints have been extracted."""
    if not isinstance(schema, dict):
        return schema
    
    if isinstance(schema, list):
        return [remove_unsupported_keywords(item, False) for item in schema]
    
    result = {}
    for key, value in schema.items():
        if not inside_properties and key in UNSUPPORTED_KEYWORDS:
            continue
        
        if isinstance(value, dict):
            if key == "properties":
                props_result = {}
                for prop_name, prop_schema in value.items():
                    props_result[prop_name] = remove_unsupported_keywords(prop_schema, False)
                result[key] = props_result
            else:
                result[key] = remove_unsupported_keywords(value, False)
        elif isinstance(value, list):
            result[key] = [remove_unsupported_keywords(item, False) for item in value]
        else:
            result[key] = value
    
    return result


def add_empty_schema_placeholder(schema: Any) -> Any:
    """Add placeholder property for empty object schemas."""
    if not isinstance(schema, dict):
        return schema
    
    if isinstance(schema, list):
        return [add_empty_schema_placeholder(item) for item in schema]
    
    result = {**schema}
    
    # Check if this is an empty object schema
    is_object_type = result.get("type") == "object"
    
    if is_object_type:
        properties = result.get("properties", {})
        has_properties = isinstance(properties, dict) and len(properties) > 0
        
        if not has_properties:
            result["properties"] = {
                EMPTY_SCHEMA_PLACEHOLDER_NAME: {
                    "type": "boolean",
                    "description": EMPTY_SCHEMA_PLACEHOLDER_DESCRIPTION,
                }
            }
            result["required"] = [EMPTY_SCHEMA_PLACEHOLDER_NAME]
    
    # Recursively process nested objects
    for key, value in list(result.items()):
        if isinstance(value, (dict, list)):
            result[key] = add_empty_schema_placeholder(value)
    
    return result


def clean_json_schema_for_antigravity(schema: Any) -> Any:
    """
    Clean a JSON schema for Antigravity API compatibility.
    
    Transforms unsupported features into description hints while preserving
    semantic information.
    """
    if not isinstance(schema, dict):
        return schema
    
    result = schema
    
    # Phase 1: Convert and add hints
    result = convert_refs_to_hints(result)
    result = convert_const_to_enum(result)
    result = add_enum_hints(result)
    
    # Phase 2: Flatten complex structures
    result = merge_all_of(result)
    result = flatten_any_of_one_of(result)
    
    # Phase 3: Cleanup
    result = remove_unsupported_keywords(result)
    
    # Phase 4: Add placeholder for empty object schemas
    result = add_empty_schema_placeholder(result)
    
    return result


def format_parameter_signature(properties: dict[str, Any], required: list[str] | None = None) -> str:
    """Format parameter signature for tool description injection."""
    if not properties:
        return ""
    
    required_set = set(required or [])
    parts = []
    
    for name, prop in properties.items():
        prop_type = prop.get("type", "any")
        
        # Handle array types
        if prop_type == "array":
            items = prop.get("items", {})
            item_type = items.get("type", "any")
            prop_type = f"ARRAY_OF_{item_type.upper()}"
            if item_type == "object" and "properties" in items:
                # Summarize object properties
                item_props = items["properties"]
                item_parts = []
                for item_name, item_prop in item_props.items():
                    item_parts.append(f"{item_name}: {item_prop.get('type', 'any')}")
                prop_type = f"ARRAY_OF_OBJECTS[{', '.join(item_parts)}]"
        
        # Handle enum types
        if "enum" in prop and isinstance(prop["enum"], list):
            enum_count = len(prop["enum"])
            prop_type = f"{prop_type} ENUM[{enum_count} options]"
        
        # Mark required
        if name in required_set:
            parts.append(f"{name} ({prop_type}, REQUIRED)")
        else:
            parts.append(f"{name} ({prop_type})")
    
    return ", ".join(parts)