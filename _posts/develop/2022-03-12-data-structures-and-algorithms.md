---
layout: post
title:  "Data Structures and Algorithms"
date:   2022-02-13
image: "assets/images/cover/surface-R8bY83YDXnY-unsplash.jpg"
categories: java dsa
---

In this blog, we will learn about the important data structures and algorithms
and also solve example problems. 

### Disjoint Sets

Disjoint Set data structure is a collection of disjoint sets where each set is
represented by one of its members. We mainly perform two operations - `union(a,b)` and
`find(a)`.

1. `union(a,b)` - This operation unions the sets of *a* and *b* and the elements in the union will be 
represented by a representative in the union.
2. `find(a)` - This operation returns the representative element of the set containing *a*.

```java
import java.util.HashMap;
import java.util.Map;

public class DisjointSet<T> {
    private final int[] rank;
    private final int[] parent;
    private final T[] elements;
    private final Map<T, Integer> elementsIndexMap;

    public DisjointSet(T[] elements) {
        this.elements = elements;
        elementsIndexMap = new HashMap<>();
        parent = new int[elements.length];
        rank = new int[elements.length];
        for (int i = 0; i < elements.length; i++) {
            if (elementsIndexMap.containsKey(elements[i]))
                throw new IllegalArgumentException("No duplicate elements allowed");
            elementsIndexMap.put(elements[i], i);
            parent[i] = i;
            rank[i] = 0;
        }

    }

    public void union(T element1, T element2) {
        T element1Representative = find(element1), element2Representative = find(element2);
        if (element1Representative.equals(element2Representative)) return;
        int element1RepIndex = elementsIndexMap.get(element1Representative);
        int element2RepIndex = elementsIndexMap.get(element2Representative);
        if (rank[element1RepIndex] < rank[element2RepIndex]) parent[element1RepIndex] = element2RepIndex;
        else if (rank[element2RepIndex] < rank[element1RepIndex]) parent[element2RepIndex] = element1RepIndex;
        else {
            parent[element2RepIndex] = element1RepIndex;
            rank[element1RepIndex]++;
        }
    }

    public T find(T element) {
        Integer index = elementsIndexMap.get(element);
        if (null == index)
            throw new IllegalArgumentException("Element not found");
        if (index == parent[index]) return element;
        T representative = find(elements[parent[index]]);
        parent[index] = elementsIndexMap.get(representative);
        return representative;
    }
}
```