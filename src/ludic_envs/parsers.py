def extract_tag_value(xml: str, tag: str) -> str:
    open_tag = f"<{tag}>"
    close_tag = f"</{tag}>"
    start = xml.find(open_tag)
    end = xml.find(close_tag, start)
    if start == -1 or end == -1:
        raise ValueError(f"Tag <{tag}> not found or malformed.")
    return xml[start + len(open_tag):end]
