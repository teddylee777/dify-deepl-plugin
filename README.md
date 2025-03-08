# DeepL Translator Plugin for Dify

A plugin for the Dify platform that provides text translation capabilities using the DeepL API.

## Features

- Translate text between multiple languages
- Automatic source language detection
- High-quality translations powered by DeepL's neural machine translation technology
- Simple integration with Dify workflows

## Installation

1. Visit your Dify platform's plugin management page
2. Select "Install from GitHub"
3. Enter the repository URL: `https://github.com/your-username/dify-deepl-plugin`
4. Select the latest version
5. Click "Install"

## Configuration

After installation, you'll need to provide your DeepL API key:

1. Go to the Plugins section in your Dify dashboard
2. Find the DeepL Translator Plugin
3. Click on "Configure"
4. Enter your DeepL API key in the credentials section
5. Save your configuration

You can obtain a DeepL API key by signing up at [DeepL API](https://www.deepl.com/pro#developer).

## Usage

Once configured, you can use the DeepL Translator in your Dify workflows:

1. Create or edit a workflow
2. Add a tool node
3. Select the DeepL Translator from the tools list
4. Configure the translation parameters:
   - `query`: The text to translate
   - `target_lang`: The target language code (e.g., "EN", "FR", "JA")
   - `source_lang`: (Optional) The source language code. If not provided, it will be auto-detected

## Supported Languages

The plugin supports all languages available through the DeepL API, including but not limited to:

- English (EN)
- German (DE)
- French (FR)
- Spanish (ES)
- Italian (IT)
- Dutch (NL)
- Polish (PL)
- Portuguese (PT)
- Russian (RU)
- Japanese (JA)
- Chinese (ZH)
- Korean (KO)

## Privacy

For information about how this plugin handles data, please see the [Privacy Policy](PRIVACY.md).

## License

[MIT License](LICENSE)

## Contact

For questions, issues, or contributions, please [open an issue](https://github.com/your-username/dify-deepl-plugin/issues) on the GitHub repository.

---

### 참고 자료

- [DeepL API 문서](https://developers.deepl.com/docs/resources/supported-languages)
- [Dify 플러그인 개발 가이드](https://docs.dify.ai/v/ko/advanced/plugins-development)



