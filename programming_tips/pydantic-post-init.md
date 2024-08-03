---
layout: post
title: "Pydantic Post Init"
modified:
categories: python
excerpt: Workaround for pydantic v1 post_init

date: 2023-01-03
share: true
ads: true
---

```python
import pydantic

class HyperParameters(pydantic.BaseModel):
    batch_size: int = 32

    @property
    def upload_batch_counts(self):
        return 1_000_000 // self.batch_size

    class Config: # will not allow undefined fields
        extra = pydantic.Extra.forbid
```

If you want to update an existing property, you can do this:

```python
class User(BaseModel):
    name: str
    age: int

    def __init__(self, **data):
        if data.get("name") == "Charlie":
            data["age"] = data.get("age", 0) + 30  # increment age by 30 if name is "Charlie"
        super().__init__(**data)
```
