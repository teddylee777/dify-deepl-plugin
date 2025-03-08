from collections.abc import Generator
from typing import Any, Dict, List, Optional
import logging

import deepl
from pydantic import BaseModel, Field
from dify_plugin import Tool, ToolProvider
from dify_plugin.entities.tool import ToolInvokeMessage
from dify_plugin.errors.tool import ToolProviderCredentialValidationError

logger = logging.getLogger(__name__)


# Simple exception class
class DeepLError(Exception):
    """Base exception class for DeepL API related errors"""

    pass


# Model for parameter validation
class ToolParameters(BaseModel):
    query: str = Field(description="Text to translate")
    target_lang: str = Field(default="KO", description="Target language code")
    source_lang: Optional[str] = Field(
        default=None, description="Source language code (auto-detection is None)"
    )


class DeeplTranslatorTool(Tool):
    """
    Tool for translating text using the DeepL API

    This tool translates various languages using the DeepL API.
    If the source language is not specified, it will be detected automatically.
    The result is returned in the form of translated text.
    """

    def _invoke(
        self, tool_parameters: dict[str, Any]
    ) -> Generator[ToolInvokeMessage, None, None]:
        """
        Method to invoke the DeepL translation tool

        Args:
            tool_parameters: Translation parameters

        Yields:
            Translation result message
        """
        # Parsing parameters
        try:
            params = ToolParameters(**tool_parameters)
        except Exception as e:
            error_msg = f"Error parsing parameters: {str(e)}"
            logger.error(error_msg)
            yield self.create_text_message(error_msg)
            return

        query = params.query
        target_lang = params.target_lang
        source_lang = params.source_lang

        # Text validation
        if not query or not query.strip():
            error_msg = "The text to translate is empty."
            logger.error(error_msg)
            yield self.create_text_message(error_msg)
            return

        # Retrieve API key
        auth_key = self.runtime.credentials["deepl_api_key"]

        try:
            # Perform translation
            translator = deepl.Translator(auth_key)
            source_lang_display = source_lang if source_lang else "Auto-detect"

            # Translation start message
            yield self.create_text_message(
                f"Translating from '{source_lang_display}' to '{target_lang}'..."
            )

            # Execute translation
            result = translator.translate_text(
                query, target_lang=target_lang, source_lang=source_lang
            )

            # Return translation result
            yield self.create_text_message(result.text)

        except Exception as e:
            error_msg = f"Error occurred during translation: {str(e)}"
            logger.error(error_msg)
            yield self.create_text_message(error_msg)


class DeeplTranslatorToolProvider(ToolProvider):
    """
    DeepL Translator Tool Provider

    This class registers and manages the DeeplTranslatorTool.
    """

    def __init__(self):
        """Initialize the DeepL Translator Tool Provider"""
        super().__init__()

    def get_tools(self) -> List[Tool]:
        """
        Return a list of available tools

        Returns:
            List[Tool]: List of available tools
        """
        return [DeeplTranslatorTool()]

    def validate_credentials(self, credentials: Dict[str, Any]) -> bool:
        """
        Validate the credentials

        Args:
            credentials (Dict[str, Any]): Credentials to validate

        Returns:
            bool: True if valid, False otherwise
        """
        return "deepl_api_key" in credentials and credentials["deepl_api_key"] != ""

    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        """
        Validate the credentials by testing the connection with the provided API key.

        Args:
            credentials: Credentials to validate

        Raises:
            ToolProviderCredentialValidationError: Invalid credentials
        """
        try:
            api_key = credentials.get("deepl_api_key", None)
            if api_key is not None:
                translator = deepl.Translator(api_key)
                # Perform a simple translation test to verify API key validity
                translator.translate_text("Hello, world!", target_lang="FR")
            else:
                raise ToolProviderCredentialValidationError(
                    "DeepL API key is not found"
                )
        except Exception as e:
            raise ToolProviderCredentialValidationError(str(e))
