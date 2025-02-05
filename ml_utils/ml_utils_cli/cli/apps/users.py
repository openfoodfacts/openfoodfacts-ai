from typing import Annotated

import typer

from ..config import LABEL_STUDIO_DEFAULT_URL

app = typer.Typer()

# Label Studio user management


@app.command()
def list(
    api_key: Annotated[str, typer.Option(envvar="LABEL_STUDIO_API_KEY")],
    label_studio_url: str = LABEL_STUDIO_DEFAULT_URL,
):
    """List all users in Label Studio."""
    from label_studio_sdk.client import LabelStudio

    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)

    for user in ls.users.list():
        print(f"{user.id:02d}: {user.email}")


@app.command()
def delete(
    user_id: int,
    api_key: Annotated[str, typer.Option(envvar="LABEL_STUDIO_API_KEY")],
    label_studio_url: str = LABEL_STUDIO_DEFAULT_URL,
):
    """Delete a user from Label Studio."""
    from label_studio_sdk.client import LabelStudio

    ls = LabelStudio(base_url=label_studio_url, api_key=api_key)
    ls.users.delete(user_id)
