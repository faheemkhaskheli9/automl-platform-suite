from django.contrib.auth.decorators import login_required
from django.shortcuts import render

from .registry import all_features


@login_required
def home(request):
    """Feature-picker dashboard home (issue #4). Gated by @login_required
    so an unauthenticated request is redirected to LOGIN_URL rather than
    ever rendering the feature list."""
    return render(request, "dashboard/home.html", {"features": all_features()})
