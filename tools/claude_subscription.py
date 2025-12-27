"""
Claude Subscription Client
===========================

Uses your existing Claude Code subscription ($20/month) via the CLI.
The CLI handles OAuth authentication internally - no API key needed.

Usage:
    from tools.claude_subscription import ClaudeClient

    client = ClaudeClient()
    response = client.chat("Analyze this trading signal...")

Prerequisites:
    1. Install CLI: npm install -g @anthropic-ai/claude-code
    2. Login once: claude login (opens browser)
    3. Verify: claude -p "test" --output-format text
"""
import json
import subprocess
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, Iterator


@dataclass
class SubscriptionInfo:
    """Claude subscription information from credentials."""
    subscription_type: str
    rate_limit_tier: str
    expires_at: int

    @classmethod
    def from_credentials(cls) -> Optional["SubscriptionInfo"]:
        """Load subscription info from credentials file."""
        creds_path = Path.home() / ".claude" / ".credentials.json"
        if not creds_path.exists():
            return None

        try:
            data = json.loads(creds_path.read_text())
            oauth = data.get("claudeAiOauth", {})
            return cls(
                subscription_type=oauth.get("subscriptionType", "unknown"),
                rate_limit_tier=oauth.get("rateLimitTier", "unknown"),
                expires_at=oauth.get("expiresAt", 0),
            )
        except Exception:
            return None


class ClaudeClient:
    """
    Claude API client using your subscription via CLI.

    This uses the Claude CLI's -p flag for non-interactive mode.
    The CLI handles OAuth authentication internally, so no API key needed.
    Your subscription usage is billed through your existing plan.
    """

    def __init__(self, timeout: int = 300, model: str = "sonnet"):
        """
        Initialize client.

        Args:
            timeout: Command timeout in seconds (default: 5 minutes)
            model: Model to use - "sonnet", "opus", or full model name
                   (default: sonnet - fast and capable for trading)
        """
        self.claude_path = shutil.which("claude")
        if not self.claude_path:
            raise FileNotFoundError(
                "Claude CLI not found.\n"
                "Install: npm install -g @anthropic-ai/claude-code\n"
                "Then login: claude login"
            )

        self.timeout = timeout
        self.model = model
        self._subscription = SubscriptionInfo.from_credentials()

        if self._subscription:
            print(f"[Claude] Subscription: {self._subscription.subscription_type}")
            print(f"[Claude] Rate limit: {self._subscription.rate_limit_tier}")
        else:
            print("[Claude] Using CLI (credentials not readable)")

    def chat(
        self,
        message: str,
        system: Optional[str] = None,
        output_format: str = "text",
        max_turns: int = 1,
    ) -> str:
        """
        Send a chat message and get response.

        Args:
            message: User message
            system: System prompt (prepended to message)
            output_format: Output format (text, json, stream-json)
            max_turns: Maximum conversation turns (default: 1 for single response)

        Returns:
            Assistant response text
        """
        # Combine system prompt with message if provided
        prompt = f"{system}\n\n{message}" if system else message

        cmd = [
            self.claude_path,
            "-p", prompt,
            "--model", self.model,
            "--output-format", output_format,
            "--max-turns", str(max_turns),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )

        if result.returncode != 0:
            error_msg = result.stderr.strip() or "Unknown error"
            raise RuntimeError(f"Claude CLI error: {error_msg}")

        return result.stdout.strip()

    def chat_json(
        self,
        message: str,
        system: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send a chat message and get JSON response.

        Args:
            message: User message
            system: System prompt

        Returns:
            Parsed JSON response
        """
        response = self.chat(message, system=system, output_format="json")
        return json.loads(response)

    def chat_stream(
        self,
        message: str,
        system: Optional[str] = None,
    ) -> Iterator[str]:
        """
        Stream a chat response.

        Yields text chunks as they arrive.
        """
        prompt = f"{system}\n\n{message}" if system else message

        cmd = [
            self.claude_path,
            "-p", prompt,
            "--model", self.model,
            "--output-format", "stream-json",
            "--max-turns", "1",
        ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            for line in process.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get("type") == "assistant":
                        # Extract text content
                        message_data = data.get("message", {})
                        content = message_data.get("content", [])
                        for block in content:
                            if block.get("type") == "text":
                                yield block.get("text", "")
                except json.JSONDecodeError:
                    continue
        finally:
            process.wait()

    def analyze_signal(
        self,
        signal_data: Dict[str, Any],
        context: str = "",
    ) -> Dict[str, Any]:
        """
        Analyze a trading signal using Claude.

        Args:
            signal_data: Signal data dict
            context: Additional context

        Returns:
            Analysis result
        """
        system = """You are a quantitative trading analyst for the Sovereign Engine.
Analyze trading signals and provide:
1. Signal quality assessment (1-10)
2. Risk factors
3. Recommended position size adjustment
4. Confidence level

Respond in JSON format only, no markdown."""

        message = f"""Analyze this trading signal:

{json.dumps(signal_data, indent=2)}

Context: {context}

Provide your analysis in JSON format with keys:
- quality_score (1-10)
- risk_factors (list)
- size_adjustment (float, 0.5-2.0)
- confidence (0-1)
- reasoning (string)"""

        try:
            response = self.chat(message, system=system)

            # Parse JSON from response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except (json.JSONDecodeError, RuntimeError) as e:
            return {"error": str(e), "raw_response": response if 'response' in dir() else None}

        return {"raw_response": response}

    def get_usage(self) -> Dict[str, Any]:
        """Get subscription info."""
        if self._subscription:
            return {
                "subscription": self._subscription.subscription_type,
                "rate_limit": self._subscription.rate_limit_tier,
                "expires_at": self._subscription.expires_at,
            }
        return {"subscription": "unknown", "note": "credentials not readable"}


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing Claude Subscription Client (CLI-based)...")
    print("=" * 50)

    try:
        client = ClaudeClient()

        # Show subscription info
        usage = client.get_usage()
        print(f"\nSubscription: {usage.get('subscription', 'unknown')}")
        print(f"Rate Limit: {usage.get('rate_limit', 'unknown')}")

        # Quick test
        print("\nSending test message...")
        response = client.chat(
            "Say 'Sovereign Engine connected' in exactly 3 words.",
        )
        print(f"Response: {response}")

        print("\n" + "=" * 50)
        print("SUCCESS - Your subscription is working!")

    except FileNotFoundError as e:
        print(f"ERROR: {e}")
    except RuntimeError as e:
        print(f"ERROR: {e}")
    except Exception as e:
        print(f"ERROR: {e}")
