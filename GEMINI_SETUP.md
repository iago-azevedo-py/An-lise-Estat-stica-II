# Instruções para configuração da API Key do Google Gemini

## Para desenvolvimento local:

1. Crie/edite o arquivo `.streamlit/secrets.toml` com sua chave:
```toml
GEMINI_API_KEY = "sua_chave_api_real"
```

2. Nunca commite esse arquivo no GitHub (já está no .gitignore)

## Para produção no Streamlit Cloud:

1. Acesse seu app no https://share.streamlit.io
2. Vá em "Settings" → "Secrets"
3. Adicione:
```toml
GEMINI_API_KEY = "sua_chave_api_real"
```

## Como obter sua API Key:

1. Acesse https://makersuite.google.com/app/apikey
2. Clique em "Create API Key"
3. Copie a chave gerada
4. Use nas configurações acima

## Segurança:
- ✅ A chave fica protegida nos secrets
- ✅ Não aparece no código fonte
- ✅ Ativação automática para usuários
- ✅ Fallback para input manual se necessário
