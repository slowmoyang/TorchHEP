import dataclasses
import inspect
from dataclasses import dataclass
from dataclasses import asdict
from dataclasses import MISSING
import argparse
import json
import typing
from colorama import Fore
# TODO Union

# TODO class _Serializable
# TODO add_argument_group

class Colors:
    """ ANSI color codes """
    BLACK = "\033[0;30m"
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    BROWN = "\033[0;33m"
    BLUE = "\033[0;34m"
    PURPLE = "\033[0;35m"
    CYAN = "\033[0;36m"
    LIGHT_GRAY = "\033[0;37m"
    DARK_GRAY = "\033[1;30m"
    LIGHT_RED = "\033[1;31m"
    LIGHT_GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    LIGHT_BLUE = "\033[1;34m"
    LIGHT_PURPLE = "\033[1;35m"
    LIGHT_CYAN = "\033[1;36m"
    LIGHT_WHITE = "\033[1;37m"
    BOLD = "\033[1m"
    FAINT = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"

class ColorfulHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):

    def _format_action_invocation(self, action):
        """https://github.com/python/cpython/blob/v3.9.13/Lib/argparse.py#L551-L573"""
        if not action.option_strings:
            default = self._get_default_metavar_for_positional(action)
            metavar, = self._metavar_formatter(action, default)(1)
            return metavar
        else:
            parts = []
            if action.nargs == 0:
                option_strings = [f'{Colors.GREEN}{each}{Fore.RESET}' for each in action.option_strings]
                parts.extend(option_strings)
            else:
                default = self._get_default_metavar_for_optional(action)
                args_string = self._format_args(action, default)
                for option_string in action.option_strings:
                    parts.append(f'{Colors.GREEN}{option_string}{Fore.RESET} {Colors.DARK_GRAY}{args_string}{Fore.RESET}')

            return ', '.join(parts)

    def _get_help_string(self, action):
        help_str = action.help
        if isinstance(help_str, str):
            if '%(default)' not in help_str:
                if action.default is not argparse.SUPPRESS:
                    defaulting_nargs = [argparse.OPTIONAL, argparse.ZERO_OR_MORE]
                    if action.option_strings or action.nargs in defaulting_nargs:
                        help_str += f' (default: {Colors.RED}%(default)s{Fore.RESET})'
        return help_str


@dataclass
class ConfigBase:
    ###########################################################################
    # post init
    ###########################################################################
    def post_init(self):
        ...

    def __post_init__(self):
        post_init_list = self._gather_post_init(self.__class__)

        # remove duplicates while prserving order
        seen = set()
        is_seen = lambda item: item in seen or seen.add(item)
        post_init_list = [each for each in post_init_list if not is_seen(each)]

        post_init_list = reversed(post_init_list)
        for post_init in post_init_list:
            post_init(self)

    @classmethod
    def _gather_post_init(cls, config_cls):
        post_init_list = []
        for each in config_cls.mro():
            if each in [ConfigBase, object]:
                continue
            for field in dataclasses.fields(each):
                if cls.is_config_class(field.type):
                    post_init_list += cls._gather_post_init(field.type)
            post_init_list.append(each.post_init)
        return post_init_list

    ###########################################################################
    #
    ###########################################################################
    @classmethod
    def from_args(cls, args=None):
        parser = argparse.ArgumentParser(formatter_class=ColorfulHelpFormatter)
        cls._add_argument_from_config_cls(parser, cls, prefixes=[])
        args = parser.parse_args(args)
        config_dict = cls._convert_namespace_to_dict(args)
        return cls.from_dict(config_dict)

    @classmethod
    def _convert_namespace_to_dict(cls, args):
        args = vars(args)
        config = {}
        for key, value in args.items():
            *namespace, name = key.split('.')
            inner = config
            while len(namespace) > 0:
                each = namespace.pop(0)
                if each not in inner:
                    inner[each] = {}
                inner = inner[each]
            inner[name] = value
        return config

    @classmethod
    def _add_argument_from_config_cls(cls, parser, config_cls, prefixes):
        for field in dataclasses.fields(config_cls):
            if issubclass(field.type, ConfigBase):
                new_prefixes = prefixes + [field.name]
                cls._add_argument_from_config_cls(parser, field.type, new_prefixes)
            else:
                cls._add_argument(parser, field, prefixes)

    @staticmethod
    def _get_metadata(field, key, default=None):
        if key in field.metadata:
            return field.metadata[key]
        else:
            return default

    @classmethod
    def _add_argument(cls, parser, field, prefixes):
        origin = typing.get_origin(field.type)

        if origin is None:
            if field.type is bool:
                cls._add_argument_bool(parser, field, prefixes)
            else:
                cls._add_argument_default(parser, field, prefixes)
        else:
            if origin is tuple:
                cls._add_argument_tuple(parser, field, prefixes)
            elif origin is list:
                cls._add_argument_list(parser, field, prefixes)
            else:
                cls._add_argument_default(parser, field, prefixes)

    @staticmethod
    def _get_flag(prefixes, field):
        name = field.name.replace('_', '-')
        if len(prefixes) > 0:
            name = '.'.join(prefixes + [name])
        flag = f'--' + name
        return flag

    @staticmethod
    def _get_default(field):
        if field.default is not MISSING:
            default = field.default
        elif field.default_factory is not MISSING:
            default = field.default_factory()
        else:
            default = None
        return default

    @classmethod
    def _add_argument_default(cls, parser, field, prefixes):
        flag = cls._get_flag(prefixes, field)
        default = cls._get_default(field)
        type = field.type
        help = cls._get_metadata(field, 'help', f'type={type.__name__}')
        return parser.add_argument(flag, default=default, type=type, help=help)

    @classmethod
    def _add_argument_tuple(cls, parser, field, prefixes):
        """requires a homogeneous sequence with a fixed size
        TODO help
        """
        args = typing.get_args(field.type)
        assert len(set(args)) == 1, args
        type = args[0]
        nargs = len(args)

        flag = cls._get_flag(prefixes, field)
        default = cls._get_default(field)
        help = cls._get_metadata(field, 'help', f'type={field.type}')
        return parser.add_argument(flag, default=default, type=type, nargs=nargs, help=help)

    @classmethod
    def _add_argument_list(cls, parser, field, prefixes):
        """requires a homogeneous sequence with a variable size
        TODO help
        """
        args = typing.get_args(field.type)
        assert len(args) == 1, args
        type = args[0]

        flag = cls._get_flag(prefixes, field)
        default = cls._get_default(field)
        help = cls._get_metadata(field, 'help', f'type={field.type}')
        return parser.add_argument(flag, default=default, type=type, nargs='+', help=help)


    @classmethod
    def _add_argument_bool(cls, parser, field, prefixes):
        assert field.type is bool
        assert field.default is not None

        flag = cls._get_flag(prefixes, field)
        action = 'store_false' if field.default else 'store_true'
        help = cls._get_metadata(field, 'help', action)
        return parser.add_argument(flag, action=action, help=help)

    @classmethod
    def from_dict(cls, config):
        for field in dataclasses.fields(cls):
            if cls.is_config_class(field.type):
                config[field.name] = field.type.from_dict(config[field.name])
        return cls(**config)

    def to_json(self, path):
        with open(path, 'w') as json_file:
            json.dump(asdict(self), json_file, indent=4)

    @classmethod
    def from_json(cls, path):
        with open(path, 'r') as json_file:
            config = json.load(json_file)
        return cls.from_dict(config)

    @staticmethod
    def is_config_class(test_cls):
        return inspect.isclass(test_cls) and issubclass(test_cls, ConfigBase)
