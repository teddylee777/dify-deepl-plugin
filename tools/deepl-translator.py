import logging
import asyncio
import hashlib
from collections.abc import Generator, AsyncGenerator
from typing import Any, List, Optional, Dict
import nest_asyncio
import aiohttp
from pydantic import BaseModel, Field
from dify_plugin import Tool
from dify_plugin.entities.tool import ToolInvokeMessage

import deepl

logger = logging.getLogger(__name__)


class DeepLError(Exception):
    """Base exception class for DeepL API errors."""

    pass


class DeepLConnectionError(DeepLError):
    """API connection error."""

    pass


class DeepLProcessingError(DeepLError):
    """Translation processing error."""

    pass


class DeepLTimeoutError(DeepLError):
    """Request timeout error."""

    pass


# Class defining the language codes supported by DeepL
class DeepLLanguages:
    # Source languages (original text languages)
    SOURCE_LANGUAGES = {
        "AR",
        "BG",
        "CS",
        "DA",
        "DE",
        "EL",
        "EN",
        "ES",
        "ET",
        "FI",
        "FR",
        "HU",
        "ID",
        "IT",
        "JA",
        "KO",
        "LT",
        "LV",
        "NB",
        "NL",
        "PL",
        "PT",
        "RO",
        "RU",
        "SK",
        "SL",
        "SV",
        "TR",
        "UK",
        "ZH",
    }

    # Target languages (translated text languages)
    TARGET_LANGUAGES = {
        "AR",
        "BG",
        "CS",
        "DA",
        "DE",
        "EL",
        "EN-GB",
        "EN-US",
        "ES",
        "ET",
        "FI",
        "FR",
        "HU",
        "ID",
        "IT",
        "JA",
        "KO",
        "LT",
        "LV",
        "NB",
        "NL",
        "PL",
        "PT-BR",
        "PT-PT",
        "RO",
        "RU",
        "SK",
        "SL",
        "SV",
        "TR",
        "UK",
        "ZH-HANS",
        "ZH-HANT",
    }

    @staticmethod
    def normalize_language_code(lang_code: str) -> str:
        """
        Normalize the language code (convert to uppercase).

        Args:
            lang_code: The language code to normalize.

        Returns:
            The normalized language code.
        """
        if not lang_code:
            return lang_code
        return lang_code.upper()

    @staticmethod
    def is_valid_source_language(lang_code: str) -> bool:
        """
        Check if the language code is a valid source language.

        Args:
            lang_code: The language code to check.

        Returns:
            True if the source language code is valid, otherwise False.
        """
        if not lang_code:
            return True  # None indicates auto-detection, which is valid
        normalized = DeepLLanguages.normalize_language_code(lang_code)
        return normalized in DeepLLanguages.SOURCE_LANGUAGES

    @staticmethod
    def is_valid_target_language(lang_code: str) -> bool:
        """
        Check if the language code is a valid target language.

        Args:
            lang_code: The language code to check.

        Returns:
            True if the target language code is valid, otherwise False.
        """
        if not lang_code:
            return False  # Target language code is required
        normalized = DeepLLanguages.normalize_language_code(lang_code)
        return normalized in DeepLLanguages.TARGET_LANGUAGES


class ToolParameters(BaseModel):
    query: str = Field(description="Text to translate")
    target_lang: str = Field(default="KO", description="Target language code")
    source_lang: Optional[str] = Field(
        default=None, description="Source language code (None indicates auto-detection)"
    )
    timeout: int = Field(default=30, description="API request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum number of retries")
    use_cache: bool = Field(default=True, description="Whether to use caching")


class DeepLTranslator:
    """Class for handling text translation using the DeepL API."""

    def __init__(self, auth_key: str, timeout: int = 30, max_retries: int = 3):
        """
        Initialize the DeepLTranslator.

        Args:
            auth_key: DeepL API authentication key.
            timeout: API request timeout in seconds.
            max_retries: Maximum number of retries.
        """
        self.auth_key = auth_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.client = deepl.Translator(auth_key)
        # Dictionary for caching translation results
        self._translation_cache: Dict[str, str] = {}

    def _get_text_hash(
        self, text: str, target_lang: str, source_lang: Optional[str] = None
    ) -> str:
        """Compute the hash value for the text and language parameters."""
        key = f"{text}:{target_lang.upper()}:{source_lang.upper() if source_lang else 'auto'}"
        hash_md5 = hashlib.md5(key.encode("utf-8"))
        return hash_md5.hexdigest()

    def _get_cached_result(
        self, text: str, target_lang: str, source_lang: Optional[str] = None
    ) -> Optional[str]:
        """Retrieve the translation result from the cache."""
        cache_key = self._get_text_hash(text, target_lang, source_lang)
        return self._translation_cache.get(cache_key)

    def _set_cached_result(
        self,
        text: str,
        target_lang: str,
        result: str,
        source_lang: Optional[str] = None,
    ) -> None:
        """Store the translation result in the cache."""
        cache_key = self._get_text_hash(text, target_lang, source_lang)
        self._translation_cache[cache_key] = result
        # Cache size limit (maximum 100 entries)
        if len(self._translation_cache) > 100:
            # Remove the oldest cache entry
            oldest_key = next(iter(self._translation_cache))
            del self._translation_cache[oldest_key]

    def _validate_language_codes(
        self, source_lang: Optional[str], target_lang: str
    ) -> None:
        """
        Validate the language codes.

        Args:
            source_lang: Source language code (None indicates auto-detection).
            target_lang: Target language code.

        Raises:
            DeepLProcessingError: If an invalid language code is provided.
        """
        # Validate source language
        if source_lang and not DeepLLanguages.is_valid_source_language(source_lang):
            raise DeepLProcessingError(f"Invalid source language code: {source_lang}")

        # Validate target language
        if not DeepLLanguages.is_valid_target_language(target_lang):
            raise DeepLProcessingError(f"Invalid target language code: {target_lang}")

    async def translate_async(
        self,
        text: str,
        target_lang: str = "KO",
        source_lang: Optional[str] = None,
        use_cache: bool = True,
    ) -> str:
        """
        Translate the text asynchronously.

        Args:
            text: Text to translate.
            target_lang: Target language code.
            source_lang: Source language code (None for auto-detection).
            use_cache: Whether to use caching.

        Returns:
            The translated text.

        Raises:
            DeepLConnectionError: If there is an API connection error.
            DeepLProcessingError: If there is an error processing the translation.
            DeepLTimeoutError: If the request times out.
        """
        # Normalize language codes
        normalized_target_lang = DeepLLanguages.normalize_language_code(target_lang)
        normalized_source_lang = (
            DeepLLanguages.normalize_language_code(source_lang) if source_lang else None
        )

        # Validate language codes
        self._validate_language_codes(normalized_source_lang, normalized_target_lang)

        # Check cached result if caching is enabled
        if use_cache:
            cached_result = self._get_cached_result(
                text, normalized_target_lang, normalized_source_lang
            )
            if cached_result:
                logger.info(f"Retrieved translation result from cache: {text[:30]}...")
                return cached_result

        try:
            # DeepL API is synchronous by default, so run_in_executor is used to run it asynchronously
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.client.translate_text(
                    text,
                    target_lang=normalized_target_lang,
                    source_lang=normalized_source_lang,
                ),
            )

            # Store result in cache
            if use_cache:
                self._set_cached_result(
                    text, normalized_target_lang, result.text, normalized_source_lang
                )

            return result.text

        except deepl.exceptions.ConnectionException as e:
            logger.error(f"DeepL API connection error: {str(e)}")
            raise DeepLConnectionError(f"DeepL API connection error: {str(e)}")
        except deepl.exceptions.DeepLException as e:
            logger.error(f"DeepL API processing error: {str(e)}")
            raise DeepLProcessingError(f"DeepL API processing error: {str(e)}")
        except asyncio.TimeoutError as e:
            logger.error(f"DeepL API request timeout: {str(e)}")
            raise DeepLTimeoutError(f"DeepL API request timeout: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during translation: {str(e)}")
            raise DeepLProcessingError(f"Unexpected error during translation: {str(e)}")

    def translate(
        self,
        text: str,
        target_lang: str = "KO",
        source_lang: Optional[str] = None,
        use_cache: bool = True,
    ) -> str:
        """
        Translate the text synchronously.

        Args:
            text: Text to translate.
            target_lang: Target language code.
            source_lang: Source language code (None for auto-detection).
            use_cache: Whether to use caching.

        Returns:
            The translated text.

        Raises:
            DeepLConnectionError: If there is an API connection error.
            DeepLProcessingError: If there is an error processing the translation.
        """
        # Normalize language codes
        normalized_target_lang = DeepLLanguages.normalize_language_code(target_lang)
        normalized_source_lang = (
            DeepLLanguages.normalize_language_code(source_lang) if source_lang else None
        )

        # Validate language codes
        self._validate_language_codes(normalized_source_lang, normalized_target_lang)

        # Check cached result if caching is enabled
        if use_cache:
            cached_result = self._get_cached_result(
                text, normalized_target_lang, normalized_source_lang
            )
            if cached_result:
                logger.info(f"Retrieved translation result from cache: {text[:30]}...")
                return cached_result

        try:
            result = self.client.translate_text(
                text,
                target_lang=normalized_target_lang,
                source_lang=normalized_source_lang,
            )

            # Store result in cache
            if use_cache:
                self._set_cached_result(
                    text, normalized_target_lang, result.text, normalized_source_lang
                )

            return result.text

        except deepl.exceptions.ConnectionException as e:
            logger.error(f"DeepL API connection error: {str(e)}")
            raise DeepLConnectionError(f"DeepL API connection error: {str(e)}")
        except deepl.exceptions.DeepLException as e:
            logger.error(f"DeepL API processing error: {str(e)}")
            raise DeepLProcessingError(f"DeepL API processing error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during translation: {str(e)}")
            raise DeepLProcessingError(f"Unexpected error during translation: {str(e)}")


class DeeplTranslatorTool(Tool):
    """
    Tool to translate text using the DeepL API.

    This tool utilizes the DeepL API to translate text between various languages.
    If the source language is not specified, it will be auto-detected.
    The result is returned as translated text.
    """

    async def _invoke_async(
        self, tool_parameters: dict[str, Any]
    ) -> AsyncGenerator[ToolInvokeMessage, None]:
        """Asynchronous version of the invoke method."""
        nest_asyncio.apply()

        # Validate and parse parameters
        try:
            params = ToolParameters(**tool_parameters)
        except Exception as e:
            error_msg = f"Error during parameter parsing: {str(e)}"
            logger.error(error_msg)
            yield self.create_text_message(error_msg)
            return

        query = params.query
        target_lang = params.target_lang
        source_lang = params.source_lang
        timeout = 60
        max_retries = 3
        use_cache = params.use_cache

        # Validate text input
        if not query or not query.strip():
            error_msg = "Translation text is empty."
            logger.error(error_msg)
            yield self.create_text_message(error_msg)
            return

        # Validate language codes
        try:
            if source_lang and not DeepLLanguages.is_valid_source_language(source_lang):
                error_msg = f"Unsupported source language code: {source_lang}"
                logger.error(error_msg)
                yield self.create_text_message(error_msg)
                return

            if not DeepLLanguages.is_valid_target_language(target_lang):
                error_msg = f"Unsupported target language code: {target_lang}"
                logger.error(error_msg)
                yield self.create_text_message(error_msg)
                return
        except Exception as e:
            error_msg = f"Error during language code validation: {str(e)}"
            logger.error(error_msg)
            yield self.create_text_message(error_msg)
            return

        auth_key = self.runtime.credentials["deepl_api_key"]
        translator = DeepLTranslator(auth_key, timeout, max_retries)

        # Attempt translation
        retries = 0
        while retries <= max_retries:
            try:
                result = await translator.translate_async(
                    query,
                    target_lang=target_lang,
                    source_lang=source_lang,
                    use_cache=use_cache,
                )

                # Provide detailed translation completion message
                detected_lang = ""
                if not source_lang:
                    detected_lang = (
                        f" (Detected source language: {result.detected_source_lang})"
                        if hasattr(result, "detected_source_lang")
                        else ""
                    )

                yield self.create_text_message(result)
                return

            except DeepLConnectionError as e:
                retries += 1
                if retries > max_retries:
                    yield self.create_text_message(
                        f"Connection error (maximum retries exceeded): {str(e)}"
                    )
                    return

                wait_time = 2**retries  # Exponential backoff
                yield self.create_text_message(
                    f"Connection error, retrying in {wait_time} seconds... ({retries}/{max_retries})"
                )
                await asyncio.sleep(wait_time)

            except (DeepLProcessingError, DeepLTimeoutError) as e:
                yield self.create_text_message(
                    f"Error occurred during translation processing: {str(e)}"
                )
                return

            except Exception as e:
                yield self.create_text_message(f"Unexpected error occurred: {str(e)}")
                return

    def _invoke(
        self, tool_parameters: dict[str, Any]
    ) -> Generator[ToolInvokeMessage, None, None]:
        """Synchronous invocation method for the DeepL translation tool."""
        # Call asynchronous version
        async_gen = self._invoke_async(tool_parameters)

        # Collect asynchronous results synchronously
        for message in asyncio.run(self._collect_async_results(async_gen)):
            yield message

    async def _collect_async_results(self, async_gen):
        """Collect results from an asynchronous generator."""
        results = []
        async for message in async_gen:
            results.append(message)
        return results
