"""Cornserve CLI entry point."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Annotated, Any

import requests
import rich
import tyro
import yaml
from rich import box
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.spinner import Spinner
from tyro.constructors import PrimitiveConstructorSpec

from cornserve.services.gateway.models import (
    AppInvocationRequest,
    AppRegistrationRequest,
    AppRegistrationResponse,
)
from cornserve.services.gateway.app.models import AppState


try:
    GATEWAY_URL = os.environ["CORNSERVE_GATEWAY_URL"]
except KeyError:
    print(
        "Environment variable CORNSERVE_GATEWAY_URL is not set. Defaulting to http://localhost:30080.\n",
    )
    GATEWAY_URL = "http://localhost:30080"

STATE_DIR = Path.home() / ".local/state/cornserve"
STATE_DIR.mkdir(parents=True, exist_ok=True)

app = tyro.extras.SubcommandApp()


def _load_payload(args: list[str]) -> dict[str, Any]:
    """Load a literal JSON or a JSON/YAML file."""
    payload = args[0]

    # A hyphen indicates stdin
    if payload == "-":
        payload = str(sys.stdin.read().strip())
    # An actual file path
    elif Path(payload).exists():
        payload = Path(payload).read_text().strip()

    # Now, payload should be either a literal JSON or YAML string
    json_error = None
    yaml_error = None

    try:
        return json.loads(payload)
    except json.JSONDecodeError as e:
        json_error = e

    try:
        return yaml.safe_load(payload)
    except yaml.YAMLError as e:
        yaml_error = e

    # Nothing worked, raise an error
    raise ValueError(
        f"Invalid payload format. JSON failed with: '{json_error}'. YAML failed with: '{yaml_error}'",
    )


class Alias:
    """App ID aliases."""

    def __init__(self, file_path: Path = STATE_DIR / "alias.json") -> None:
        """Initialize the Alias class."""
        self.file_path = file_path
        # Alias -> App ID
        self.aliases = {}
        if file_path.exists():
            with open(file_path) as file:
                self.aliases = json.load(file)

    def get(self, alias: str) -> str | None:
        """Get the app ID for an alias."""
        return self.aliases.get(alias)

    def reverse_get(self, app_id: str) -> str | None:
        """Get the alias for an app ID."""
        for alias, id_ in self.aliases.items():
            if id_ == app_id:
                return alias
        return None

    def set(self, app_id: str, alias: str) -> None:
        """Set an alias for an app ID."""
        if alias.startswith("app-"):
            raise ValueError("Alias cannot start with 'app-'")
        self.aliases[alias] = app_id
        with open(self.file_path, "w") as file:
            json.dump(self.aliases, file)

    def remove(self, alias: str) -> None:
        """Remove an alias for an app ID."""
        self.aliases.pop(alias, None)
        with open(self.file_path, "w") as file:
            json.dump(self.aliases, file)


@app.command(name="register")
def register(
    path: Annotated[Path, tyro.conf.Positional],
    alias: str | None = None,
) -> None:
    """Register an app with the Cornserve gateway.

    Args:
        path: Path to the app's source file.
        alias: Optional alias for the app.
    """
    request = AppRegistrationRequest(source_code=path.read_text().strip())
    try:
        raw_response = requests.post(
            f"{GATEWAY_URL}/app/register",
            json=request.model_dump(),
            timeout=10,
        )
        raw_response.raise_for_status()
        response = AppRegistrationResponse.model_validate(raw_response.json())
        app_id = response.app_id
    except requests.exceptions.RequestException as e:
        rich.print(Panel(f"Failed to send registration request: {e}", style="red", expand=False))
        return
    except Exception as e:
        rich.print(Panel(f"Failed to initiate registration: {e}", style="red", expand=False))
        return

    if not app_id:
        rich.print(Panel("Failed to get app_id from registration response.", style="red", expand=False))
        return

    current_alias = alias or path.stem
    Alias().set(app_id, current_alias)

    # Initial display before polling starts
    initial_table = Table(box=box.ROUNDED)
    initial_table.add_column("App ID")
    initial_table.add_column("Alias")
    initial_table.add_column("Status")
    initial_table.add_row(app_id, current_alias, Text(AppState.NOT_READY.value.title(), style="yellow"))
    rich.print(initial_table)
    
    with Live(auto_refresh=False, vertical_overflow="visible") as live:
        status_str = AppState.NOT_READY.value
        spinner_message = f" Registering app '{app_id}'... Current status: {status_str.title()}"
        spinner = Spinner("dots", text=Text(spinner_message, style="yellow"))
        live.update(spinner, refresh=True)
        
        # Brief pause to allow the server to set the initial NOT_READY state if it's extremely fast
        # and before the first poll, though app_manager sets it before returning app_id.
        time.sleep(0.2)

        while status_str == AppState.NOT_READY.value:
            try:
                status_response = requests.get(f"{GATEWAY_URL}/app/status/{app_id}", timeout=5)
                status_response.raise_for_status()
                status_data = status_response.json()
                status_str = status_data.get("status", "unknown").lower()
                
                current_style = "yellow"
                if status_str == AppState.READY.value:
                    current_style = "green"
                elif status_str == AppState.REGISTRATION_FAILED.value:
                    current_style = "red"
                
                spinner_message = f" Registering app '{app_id}'... Current status: {status_str.title()}"
                spinner = Spinner("dots", text=Text(spinner_message, style=current_style))
                live.update(spinner, refresh=True)
                
                if status_str != AppState.NOT_READY.value:
                    break
                time.sleep(1)
            except requests.exceptions.Timeout:
                spinner_message = f" Polling timeout for app '{app_id}'. Retrying..."
                spinner = Spinner("dots", text=Text(spinner_message, style="orange"))
                live.update(spinner, refresh=True)
                time.sleep(1)
            except requests.exceptions.RequestException as e:
                status_str = "polling_error"
                live.update(Text(f"Error polling status for '{app_id}': {e}", style="red"), refresh=True)
                break 
            except Exception as e:
                status_str = "unexpected_error"
                live.update(Text(f"An unexpected error occurred while polling for '{app_id}': {e}", style="red"), refresh=True)
                break

    # Final status display outside of Live context
    final_table = Table(box=box.ROUNDED, title=f"Registration Result for {app_id}")
    final_table.add_column("App ID")
    final_table.add_column("Alias")
    final_table.add_column("Final Status")
    
    final_status_text = status_str.title()
    final_status_style = "yellow"

    if status_str == AppState.READY.value:
        final_status_style = "green"
        rich.print(Panel(f"App '{app_id}' registered successfully with alias '{current_alias}'.", style="green", expand=False))
    elif status_str == AppState.REGISTRATION_FAILED.value:
        final_status_style = "red"
        Alias().remove(current_alias)
        rich.print(Panel(f"App '{app_id}' failed to register. Alias '{current_alias}' removed.", style="red", expand=False))
    elif status_str == "polling_error":
        rich.print(Panel(f"Could not determine final status for app '{app_id}' due to polling errors. Please check with 'cornserve list'.", style="red", expand=False))
    elif status_str == "unexpected_error":
        rich.print(Panel(f"An unexpected error occurred while checking status for '{app_id}'. Please check with 'cornserve list'.", style="red", expand=False))
    else:
        rich.print(Panel(f"App '{app_id}' registration ended with an inconclusive status: '{status_str.title()}'. Please check with 'cornserve list'.", style="yellow", expand=False))

    final_table.add_row(app_id, current_alias, Text(final_status_text, style=final_status_style))
    rich.print(final_table)


@app.command(name="unregister")
def unregister(
    app_id_or_alias: Annotated[str, tyro.conf.Positional],
) -> None:
    """Unregister an app from Cornserve.

    Args:
        app_id_or_alias: ID of the app to unregister or its alias.
    """
    if app_id_or_alias.startswith("app-"):
        app_id = app_id_or_alias
    else:
        alias = Alias()
        app_id = alias.get(app_id_or_alias)
        if not app_id:
            rich.print(Panel(f"Alias {app_id_or_alias} not found.", style="red", expand=False))
            return
        alias.remove(app_id_or_alias)

    raw_response = requests.post(
        f"{GATEWAY_URL}/app/unregister/{app_id}",
    )
    if raw_response.status_code == 404:
        rich.print(Panel(f"App {app_id} not found.", style="red", expand=False))
        return

    raw_response.raise_for_status()

    rich.print(Panel(f"App {app_id} unregistered successfully.", expand=False))


@app.command(name="list")
def list_apps() -> None:
    """List all registered apps."""
    raw_response = requests.get(f"{GATEWAY_URL}/app/list")
    raw_response.raise_for_status()
    response: dict[str, str] = raw_response.json()

    alias = Alias()

    table = Table(box=box.ROUNDED)
    table.add_column("App ID")
    table.add_column("Alias")
    table.add_column("Status")
    for app_id, status in response.items():
        table.add_row(
            app_id, alias.reverse_get(app_id) or "", Text(status, style="green" if status == "ready" else "yellow")
        )
    rich.print(table)


@app.command(name="invoke")
def invoke(
    app_id_or_alias: Annotated[str, tyro.conf.Positional],
    data: Annotated[
        dict[str, Any],
        PrimitiveConstructorSpec(
            nargs=1,
            metavar="JSON|YAML",
            instance_from_str=_load_payload,
            is_instance=lambda x: isinstance(x, dict),
            str_from_instance=lambda d: [json.dumps(d)],
        ),
        tyro.conf.Positional,
    ],
) -> None:
    """Invoke an app with the given data.

    Args:
        app_id_or_alias: ID of the app to invoke or its alias.
        data: Input data for the app. This can be a literal JSON string,
            a path to either a JSON or YAML file, or a hyphen to read in from stdin.
    """
    if app_id_or_alias.startswith("app-"):
        app_id = app_id_or_alias
    else:
        alias = Alias()
        app_id = alias.get(app_id_or_alias)
        if not app_id:
            rich.print(Panel(f"Alias {app_id_or_alias} not found.", style="red", expand=False))
            return

    request = AppInvocationRequest(request_data=data)
    raw_response = requests.post(
        f"{GATEWAY_URL}/app/invoke/{app_id}",
        json=request.model_dump(),
    )

    if raw_response.status_code == 404:
        rich.print(Panel(f"App {app_id} not found.", style="red", expand=False))
        return

    raw_response.raise_for_status()

    table = Table(box=box.ROUNDED, show_header=False)
    for key, value in raw_response.json().items():
        table.add_row(key, value)
    rich.print(table)


def main() -> None:
    """Main entry point for the Cornserve CLI."""
    app.cli(description="Cornserve CLI")
