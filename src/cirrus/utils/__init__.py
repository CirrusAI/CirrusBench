# -*- coding: utf-8 -*-

# Core data models and basic utilities
from .basic_utils import (
    RunConfig,
    TimelineMessage,
    OdysseyTask,
    FunctionCall,
    ToolCall,
    RoleType,
    format_enriched_dialog,
    separate_tool_messages,
    merge_messages_with_tool,
    # JSON and data utilities
    parse_json_string,
    extract_json_body,
    load_json_dict,
    save_json_dict,
    load_json_list,
    dump_json_list,
    show_dict_keys,
    flatten_dict,
    multilayer_get_item,
    # File I/O utilities
    save_pickle,
    load_pickle,
    # Time utilities
    make_nowtime_tag,
)

# File I/O operations
from .io_utils import (
    expand_paths,
    load_file,
    dump_file,
)

# System utilities
from .utils import (
    DATA_DIR,
    get_dict_hash,
    show_dict_diff,
    get_now,
    format_time,
    get_commit_hash,
)

# Pydantic utilities
from .pydantic_utils import (
    BaseModelNoExtra,
    get_pydantic_hash,
    update_pydantic_model_with_dict,
)

# Display utilities (imported on demand to avoid circular imports with data_model)
# from .display import ConsoleDisplay, MarkdownDisplay

# Action transformation utilities (XML/JSON)
# from .action_transform_json_xml import (
#     action_transform_json_xml,
#     convert_value,
#     extract_json_from_markdown,
#     match_action_responses_fix,
#     match_action_responses_key_only,
#     match_action_response_plus,
#     use_modified_prompt,
#     push_with_stacking_when_xml_match,
# )


__all__ = [
    # Data models
    "RunConfig",
    "TimelineMessage",
    "OdysseyTask",
    "FunctionCall",
    "ToolCall",
    "RoleType",

    # Message processing
    "format_enriched_dialog",
    "separate_tool_messages",
    "merge_messages_with_tool",

    # JSON/data utilities
    "parse_json_string",
    "extract_json_body",
    "load_json_dict",
    "save_json_dict",
    "load_json_list",
    "dump_json_list",
    "show_dict_keys",
    "flatten_dict",
    "multilayer_get_item",

    # File I/O
    "expand_paths",
    "load_file",
    "dump_file",
    "save_pickle",
    "load_pickle",

    # System utilities
    "DATA_DIR",
    "get_dict_hash",
    "show_dict_diff",
    "get_now",
    "format_time",
    "get_commit_hash",
    "make_nowtime_tag",

    # Pydantic utilities
    "BaseModelNoExtra",
    "get_pydantic_hash",
    "update_pydantic_model_with_dict",

    # Action transform functions
    "action_transform_json_xml",
    "convert_value",
    "extract_json_from_markdown",
    "match_action_responses_fix",
    "match_action_responses_key_only",
    "match_action_response_plus",
    "use_modified_prompt",
    "push_with_stacking_when_xml_match",
]
