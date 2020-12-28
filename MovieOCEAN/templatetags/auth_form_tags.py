from django.template import Library

register = Library()


@register.filter(name='add_class')
def add_class(field, class_attr):
    """
    This template tag enables you to add a css class to a form field (e.g. in authentication templates)
    Example: {{ field | add_class:'form-control' }}
    """
    return field.as_widget(attrs={'class': class_attr})
