from django import template

register = template.Library()


@register.filter
def dictkey(d, key):
    """Look up `key` in dict `d` — Django templates have no `d[key]` syntax
    for a variable key, only a literal one."""
    if d is None:
        return ""
    return d.get(key, "")
