# reinforcenow/cli/main.py

import click
from reinforcenow.cli.commands import login, logout, status, orgs, start, run, stop


@click.group(context_settings={'help_option_names': ['-h', '--help']})
@click.option('--api-url', default='https://www.reinforcenow.ai/api', help='API base URL')
@click.option('--debug', is_flag=True, hidden=True)
@click.pass_context
def cli(ctx, api_url, debug):
    """Train language models with reinforcement learning."""
    ctx.ensure_object(dict)
    ctx.obj['api_url'] = api_url
    ctx.obj['debug'] = debug


# Add commands
cli.add_command(login)
cli.add_command(logout)
cli.add_command(status)
cli.add_command(orgs)
cli.add_command(start)
cli.add_command(run)
cli.add_command(stop)


def main():
    """Entry point."""
    cli(auto_envvar_prefix='REINFORCE')


if __name__ == "__main__":
    main()