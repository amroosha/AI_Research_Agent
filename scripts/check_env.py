"""Print safe environment configuration status."""

from __future__ import annotations

from research_agent.env import env_status

if __name__ == "__main__":
    for name, present in env_status().items():
        print(f"{name}: {'set' if present else 'missing'}")
