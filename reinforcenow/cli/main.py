# reinforcenow/cli/main.py

import click
from reinforcenow.cli import commands


@click.group(context_settings={'help_option_names': ['-h', '--help']})
@click.option('--api-url', default='https://www.reinforcenow.ai/api', help='API base URL')
@click.option('--debug', is_flag=True, hidden=True)
@click.pass_context
def cli(ctx, api_url, debug):
    """Train language models with reinforcement learning from human feedback."""
    ctx.ensure_object(dict)
    ctx.obj['api_url'] = api_url
    ctx.obj['debug'] = debug

    commands.api.base_url = api_url


cli.add_command(commands.login)
cli.add_command(commands.logout)
cli.add_command(commands.status)
cli.add_command(commands.orgs)
cli.add_command(commands.start)
cli.add_command(commands.run)
cli.add_command(commands.stop)


def main():
    """Entry point."""
    cli(auto_envvar_prefix='REINFORCE')


if __name__ == "__main__":
    main()