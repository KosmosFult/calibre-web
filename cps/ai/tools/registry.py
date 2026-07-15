from functools import wraps


class AgentTool:
    """
    Decorator used to register python functions as callable Agent tools.
    """

    _registry = {}

    def __init__(self, name, description, parameters):
        self.name = name
        self.description = description
        self.parameters = parameters

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        AgentTool._registry[self.name] = {
            "function": wrapper,
            "declaration": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
        return wrapper

    @classmethod
    def get_function_declarations(cls):
        """Return JSON-schema function declarations."""
        return [item["declaration"] for item in cls._registry.values()]

    @classmethod
    def openai_tools(cls):
        """Return tools in the OpenAI Chat Completions shape."""
        return [
            {
                "type": "function",
                "function": dict(item["declaration"]),
            }
            for item in cls._registry.values()
        ]

    @classmethod
    def get_tools_functions(cls):
        """Return the registered Python callables."""
        return [item["function"] for item in cls._registry.values()]

    @classmethod
    def get_tool_func(cls, name):
        registry_item = cls._registry.get(name)
        if registry_item:
            return registry_item["function"]
        return None


__all__ = ["AgentTool"]
