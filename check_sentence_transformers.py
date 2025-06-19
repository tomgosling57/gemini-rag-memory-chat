try:
    import sentence_transformers
    print('sentence_transformers is installed')
except ModuleNotFoundError:
    print('sentence_transformers is not installed')