"""Resource type IDs and the hash lookup tables shipped with the add-on.

The tables (``hashlists/*.txt|json``) are plain text and are reused verbatim:
they are just name<->hash lists, no code involved.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from .hashing import murmur64_hash, murmur32_hash, hash_name

__all__ = [
    "TYPE_UNIT", "TYPE_GEOMETRY_GROUP", "TYPE_TEXTURE", "TYPE_MATERIAL",
    "TYPE_BONES", "TYPE_ANIMATION", "TYPE_STATE_MACHINE", "TYPE_PARTICLE",
    "TYPE_XAML", "TYPE_STRINGS", "TYPE_PHYSICS",
    "TYPE_WWISE_BANK", "TYPE_WWISE_DEP", "TYPE_WWISE_STREAM", "TYPE_WWISE_METADATA",
    "TYPE_NAMES", "TYPE_EXTENSIONS", "TOC_MAGIC", "DSAR_MAGIC",
    "type_name", "type_extension", "NameTable", "parse_resource_id",
]

# --------------------------------------------------------------------------- #
# type ids (decimal values as used by the engine's 64 bit type hashes)
# --------------------------------------------------------------------------- #
TYPE_BONES = 0x18DEAD01056B72E9
TYPE_LEVEL = 0x2A690FD348FE9AC5
TYPE_PHYSICS = 0x5F7203C8F280DAB8
TYPE_ANIMATION = 0x931E336D7646CC26
TYPE_STATE_MACHINE = 0xA486D4045106165C
TYPE_PARTICLE = 0xA8193123526FAD64
TYPE_UNIT = 0xE0A48D0BE9A7453F
TYPE_GEOMETRY_GROUP = 0xC4F0F4BE7FB0C8D6
TYPE_TEXTURE = 0xCD4238C6A0C69E32
TYPE_MATERIAL = 0xEAC0B497876ADEDF
TYPE_STRINGS = 0x0D972BAB10B40FD3
TYPE_XAML = 0x46BC82AAE9AE0565
TYPE_WWISE_BANK = 0x535A7BD3E650D799
TYPE_WWISE_DEP = 0xAF32095C82F2B070
TYPE_WWISE_STREAM = 0x504B55235D21440E
TYPE_WWISE_METADATA = 0xD50A8B7E1C82B110
#: additional types listed in the add-on (``Global_TypeIDs``)
TYPE_PREFAB = 0xAB2F78E885F513C6
TYPE_SHADING_ENV_MAPPING = 0x250E0A11AC8E26F8
TYPE_ENTITY = 0x9831CA893B0D087D
TYPE_BIK = 0xAA5965F03029FA18
TYPE_FONT = 0x9EFE0A916AAE7880
TYPE_CLOTH = 0xD7014A50477953E0
TYPE_SHADING_ENV = 0xFE73C7DCFF8A7CA5
TYPE_LUA = 0xA14E8DFA2CD117E2
TYPE_VECTOR_FIELD = 0xF7505933166D6755
TYPE_FLOW = 0x92D3EE038EEB610D
TYPE_RENDER_CONFIG = 0x27862FE24795319C
TYPE_NETWORK_CONFIG = 0x3B1FA9E8F6BAC374
TYPE_CONFIG = 0x82645835E6B73232
TYPE_PACKAGE = 0xAD9C6D9ED1E5E77A
TYPE_MOUSE_CURSOR = 0xB277B11FE4A61D37
TYPE_RENDERABLE = 0x7910103158FC1DE9
TYPE_SHADER_LIBRARY_GROUP = 0x9E5C3CC74575AEB5
TYPE_SHADER_LIBRARY = 0xE5EE32A477239A93
TYPE_RUNTIME_FONT = 0x05106B81DCD58A13
TYPE_RAGDOLL_PROFILE = 0x1D59BD6687DB6B33
TYPE_AH_BIN = 0x2A0A70ACFE476E1D
TYPE_IK_SKELETON = 0x57A13425279979D7
TYPE_WWISE_PROPERTIES = 0x5FDD5FE391076F9F
TYPE_HAVOK_AI_PROPERTIES = 0x6592B918E67F082C
TYPE_TEXTURE_ATLAS = 0x9199BB50B6896F02
TYPE_GELETA = 0xB8FD4D2CEDE20ED7
TYPE_HASH_LOOKUP = 0xE3F2851035957AF5
TYPE_SPEEDTREE = 0xE985C5F61C169997
TYPE_HAVOK_PHYSICS_PROPERTIES = 0xF7A09F8BB35A1D49
TYPE_CAMERA_SHAKE = 0xFCAAF813B4D3CC1E
TYPE_BIK2 = 0x5EE65304478F8DB5

# Container magic numbers
TOC_MAGIC = 0xF0000011          # 4026531857  "legacy" toc/package header
DSAR_MAGIC = 0x52415344         # 1380012868  b"DSAR" chunked container

TYPE_NAMES = {
    TYPE_BONES: "bones",
    TYPE_LEVEL: "level",
    TYPE_PHYSICS: "physics",
    TYPE_ANIMATION: "animation",
    TYPE_STATE_MACHINE: "state_machine",
    TYPE_PARTICLE: "particles",
    TYPE_UNIT: "mesh",
    TYPE_GEOMETRY_GROUP: "geometry_group",
    TYPE_TEXTURE: "texture",
    TYPE_MATERIAL: "material",
    TYPE_STRINGS: "strings",
    TYPE_XAML: "xaml",
    TYPE_WWISE_BANK: "wwise_bank",
    TYPE_WWISE_DEP: "wwise_dep",
    TYPE_WWISE_STREAM: "wwise_stream",
    TYPE_WWISE_METADATA: "wwise_metadata",
    TYPE_PREFAB: "prefab",
    TYPE_SHADING_ENV_MAPPING: "shading_environment_mapping",
    TYPE_ENTITY: "entity",
    TYPE_BIK: "bik",
    TYPE_FONT: "font",
    TYPE_CLOTH: "cloth",
    TYPE_SHADING_ENV: "shading_environment",
    TYPE_LUA: "lua",
    TYPE_VECTOR_FIELD: "vector_field",
    TYPE_FLOW: "flow",
    TYPE_RENDER_CONFIG: "render_config",
    TYPE_NETWORK_CONFIG: "network_config",
    TYPE_CONFIG: "config",
    TYPE_PACKAGE: "package",
    TYPE_MOUSE_CURSOR: "mouse_cursor",
    TYPE_RENDERABLE: "renderable",
    TYPE_SHADER_LIBRARY_GROUP: "shader_library_group",
    TYPE_SHADER_LIBRARY: "shader_library",
    TYPE_RUNTIME_FONT: "runtime_font",
    TYPE_RAGDOLL_PROFILE: "ragdoll_profile",
    TYPE_AH_BIN: "ah_bin",
    TYPE_IK_SKELETON: "ik_skeleton",
    TYPE_WWISE_PROPERTIES: "wwise_properties",
    TYPE_HAVOK_AI_PROPERTIES: "havok_ai_properties",
    TYPE_TEXTURE_ATLAS: "texture_atlas",
    TYPE_GELETA: "geleta",
    TYPE_HASH_LOOKUP: "hash_lookup",
    TYPE_SPEEDTREE: "speedtree",
    TYPE_HAVOK_PHYSICS_PROPERTIES: "havok_physics_properties",
    TYPE_CAMERA_SHAKE: "camera_shake",
    TYPE_BIK2: "bik2",
}

#: file name suffix used when a resource is unpacked as a loose payload
TYPE_EXTENSIONS = {
    TYPE_BONES: ".bones",
    TYPE_LEVEL: ".level",
    TYPE_PHYSICS: ".physics",
    TYPE_ANIMATION: ".animation",
    TYPE_STATE_MACHINE: ".state_machine",
    TYPE_PARTICLE: ".particles",
    TYPE_UNIT: ".unit",
    TYPE_GEOMETRY_GROUP: ".geometry_group",
    TYPE_TEXTURE: ".texture",
    TYPE_MATERIAL: ".material",
    TYPE_STRINGS: ".strings",
    TYPE_XAML: ".xaml",
    TYPE_WWISE_BANK: ".wwise_bank",
    TYPE_WWISE_DEP: ".wwise_dep",
    TYPE_WWISE_STREAM: ".wwise_stream",
    TYPE_WWISE_METADATA: ".wwise_metadata",
    TYPE_LUA: ".lua",
    TYPE_PREFAB: ".prefab",
    TYPE_ENTITY: ".entity",
    TYPE_CONFIG: ".config",
}

#: parent material ids the add-on recognises (used for template selection)
PARENT_MATERIALS = {
    3430705909399566334:  "basic+",
    15586118709890920288: "alphaclip",
    6101987038150196875:  "original",
    15356477064658408677: "basic",
    15235712479575174153: "emissive",
    17265463703140804126: "advanced",
    17720495965476876300: "armorlut",
    9576304397847579354:  "translucent",
    8580182439406660688:  "basic+",
}


def type_name(type_id: int) -> str:
    return TYPE_NAMES.get(int(type_id), f"unknown_{int(type_id):016X}")


def type_extension(type_id: int) -> str:
    return TYPE_EXTENSIONS.get(int(type_id), f".{type_name(type_id)}")


#: bare hex ids as printed by ``srkit list`` (must contain a letter so that a pure
#: decimal id is never reinterpreted)
_BARE_HEX = re.compile(r"^(?=.*[A-Fa-f])[0-9A-Fa-f]{1,16}$")


def parse_resource_id(text: str, names: "NameTable | None" = None) -> int:
    """Accept ``123``, ``0x…``, a bare hex id as printed by ``list``, or a friendly name."""
    text = text.strip()
    if not text:
        raise ValueError("empty resource id")
    if text.lower().startswith("0x"):
        return int(text, 16)
    try:
        return int(text)
    except ValueError:
        pass
    if _BARE_HEX.match(text):
        # e.g. 12CA15252610A40A - copy/pasted straight out of `srkit list`
        return int(text, 16)
    if names is not None:
        hit = names.find_by_name(text)
        if hit is not None:
            return hit
    # otherwise: treat as a hashable name (what the engine would do)
    return murmur64_hash(text.encode("utf-8"))


@dataclass
class NameTable:
    """Friendly names / bone names / shader variables / archive names."""

    names: dict[int, str] = field(default_factory=dict)
    reverse: dict[str, int] = field(default_factory=dict)
    bones: dict[int, str] = field(default_factory=dict)
    shader_variables: dict[int, str] = field(default_factory=dict)
    types: dict[int, str] = field(default_factory=dict)
    archives: dict[int, str] = field(default_factory=dict)

    # -- lookups ------------------------------------------------------- #
    def friendly(self, resource_id: int) -> str:
        return self.names.get(int(resource_id), "")

    def bone(self, bone_hash: int) -> str:
        return self.bones.get(int(bone_hash), "")

    def shader_variable(self, var_id: int) -> str:
        return self.shader_variables.get(int(var_id), "")

    def type_name(self, type_id: int) -> str:
        return self.types.get(int(type_id)) or type_name(type_id)

    def archive_name(self, resource_id: int) -> str:
        return self.archives.get(int(resource_id), "")

    def label(self, resource_id: int) -> str:
        """Best human readable label for a resource id."""
        friendly = self.friendly(resource_id)
        if friendly:
            return friendly
        archive = self.archive_name(resource_id)
        if archive:
            return archive
        return f"{int(resource_id):016X}"

    def find_by_name(self, name: str) -> int | None:
        return self.reverse.get(name)

    def hash_name(self, name: str) -> int:
        return hash_name(name)

    # -- loading ------------------------------------------------------- #
    @classmethod
    def load(cls, hashlist_dir: str | os.PathLike) -> "NameTable":
        table = cls()
        base = os.fspath(hashlist_dir)
        if not os.path.isdir(base):
            return table

        friendly = os.path.join(base, "friendlynames.txt")
        if os.path.isfile(friendly):
            with open(friendly, "r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    parts = line.split(" ", 1)
                    if len(parts) != 2:
                        continue
                    try:
                        rid = int(parts[0])
                    except ValueError:
                        continue
                    table.names.setdefault(rid, parts[1])
                    table.reverse.setdefault(parts[1], rid)

        bone_file = os.path.join(base, "bonehash.txt")
        if os.path.isfile(bone_file):
            with open(bone_file, "r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            table.bones[int(parts[0])] = parts[1]
                        except ValueError:
                            continue

        var_file = os.path.join(base, "shadervariables.txt")
        if os.path.isfile(var_file):
            with open(var_file, "r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            table.shader_variables[int(parts[1], 16)] = parts[0]
                        except ValueError:
                            continue

        type_file = os.path.join(base, "typehash.txt")
        if os.path.isfile(type_file):
            with open(type_file, "r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            table.types[int(parts[0], 16)] = parts[1]
                        except ValueError:
                            continue

        archive_file = os.path.join(base, "archivehashes.json")
        if os.path.isfile(archive_file):
            try:
                with open(archive_file, "r", encoding="utf-8", errors="replace") as handle:
                    data = json.load(handle)
                for title, entries in data.items():
                    for key, value in entries.items():
                        try:
                            table.archives[int(key)] = f"{title}: {value}"
                        except ValueError:
                            continue
            except (json.JSONDecodeError, OSError):
                pass

        return table
