---
layout: post
title:  "Arrange GameObjects randomly in Unity"
date:   2022-02-13
image: "assets/images/cover/surface-R8bY83YDXnY-unsplash.jpg"
categories: unity brick-breaker
---

In this blog, we will learn how to randomly arrange objects within the screen boundaries
in Unity game engine. We will take **Brick Breaker** game as an example to demonstrate how to randomly
add bricks within the required screen space.

### Introduction

If you have never played a Brick Breaker game, please check this out -
[Brick Breaker](https://brick-breaker.dhirajsalian.com). For the ones who don't want to
play, here is how the game looks like.

![Brick Breaker Game]({{ 'assets/images/brick-breaker.png' | relative_url }})

This blog will cover the logic to build the generation of random brick arrangement for
every new level in the game.

Before moving ahead, if you think you are a beginner in Unity, please go through this amazing
tutorial - [Brick Breaker Game](https://www.youtube.com/watch?v=NWG8vO02oj4&ab_channel=freeCodeCamp.org).

### Brick Creation

We will be creating a Brick Prefab with the following property and scripts. The important
property to note is the scale (X=1.8 and Y=0.8). The scale is important if you want to 
have spacing between your bricks. The other important properties are point, colors, explosion
and livesPowerUp required by the BrickScript.

![Brick Properties]({{ 'assets/images/brick-properties.png' | relative_url }})

The BrickScript handles setting brick color, initializing explosion or randomly giving power
ups when ball is hit. We will not be delving deep into the code, but here is the code for your reference.

```cs
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BrickScript : MonoBehaviour
{
    public int point;

    public Color[] colors;

    public Transform explosion;

    public Transform livesPowerUp;

    GameManager gm;

    int hitsToBreak;

    SpriteRenderer brickRenderer;

    // Start is called before the first frame update
    void Start()
    {
        gm = GameObject.Find("GameManager").GetComponent<GameManager>();
        if (point > colors.Length)
        {
            point = colors.Length;
        }
        hitsToBreak = point;
        brickRenderer = gameObject.GetComponent<SpriteRenderer>();
        SetBrickColor(hitsToBreak - 1);
    }

    // Update is called once per frame
    void Update()
    {

    }

    public void BallHitUpdate()
    {
        hitsToBreak--;
        if (hitsToBreak > 0)
        {
            SetBrickColor(hitsToBreak - 1);
        }
        else
        {
            Explode();
            GivePowerUpRandomly();
            Destroy(gameObject);
        }
    }

    void Explode()
    {
        Transform newExplosion = Instantiate(explosion, transform.position, transform.rotation);
        Destroy(newExplosion.gameObject, 2.5f);
        gm.UpdateScore(point);
        gm.UpdateNumberOfBricks();
    }

    void GivePowerUpRandomly()
    {
        float random = Random.Range(1, 101);
        if (random < 10)
        {
            Instantiate(livesPowerUp, transform.position, transform.rotation);
        }
    }

    void SetBrickColor(int index)
    {
        brickRenderer.color = colors[index];
    }
}
```

### Brick Arrangement

To arrange bricks in the screen space, we need to visualize the brick container space as a 2-dimensional matrix.
The screen co-ordinates should be known to determine the brick container space co-ordinates.

#### Determining Screen Co-ordinates

```cs
Vector3 bottomLeft = mainCamera.ViewportToWorldPoint(new Vector3(0, 0, -10));
Vector3 topRight = mainCamera.ViewportToWorldPoint(new Vector3(1, 1, -10));
float screenWidth = Mathf.Floor(topRight.x - bottomLeft.x);
float screenHeight = Mathf.Floor(topRight.y - bottomLeft.y);
```

#### Determining the Brick Container Space Co-ordinates

Once we have screenWidth and screenHeight, it is easier to determine the brick container space co-ordinates.
To get the count of bricks in horizontal direction, considering the brick length as 2 units and leaving space of 1 brick
each at either side, we need to divide the screenWidth by 2 and reduce the result by 2 bricks.
```cs
int countX = (int)Mathf.Floor(screenWidth/2)-2;
```

To get the count of bricks in vertical direction, considering the brick length as 1 unit and leaving space of 1 brick
each at either side, we need to divide the screenHeight by 2 as we don't want them to appear near the bottom half of
the screen.
```cs
int countY = (int)Mathf.Floor(screenHeight/2)-2;
```

#### Determining the start Co-ordinates of Brick Container Space

To get the start co-ordinates of Brick Container for horizontal direction, we divide the countX by 2 and negate it. The
reason we divide and negate is to start building bricks from left side of 0 co-ordinate. To get the start co-ordinate
of brick container for vertical direction, we would directly consider it as 1, since we want it to be above 0 co-ordinate.
```cs
int startX = -countX/2;
int startY = 1;
```

### Random Brick Placement

The logic to place bricks randomly is simple. It is as follows:
1. Randomly generate x position, y position and points for a brick.
2. Check if brick is already present at that random position.
3. If brick not present at that random position, place the brick at that position.
4. If the brick is present at that random position, keep going to left of random position till you find empty spot for 
brick placement.
5. If the brick crosses startX, reinitialize x to initial position (random position generated at Step 1).
6. Repeat step 4 and 5 in the right direction, upward direction and bottom direction till you find an empty spot.

Note: The amount of brick to be placed should not be same as amount of positions available. A good amount of brick would
be around 40% to 75% of positions available.

```cs
void BuildNewBrickArrangement()
{
    numberOfBricks = (int)Mathf.Floor(Random.Range(Mathf.Floor(countX * countY * 0.4f), Mathf.Floor(countX * countY * 0.7f)));
    for (int i = 0; i < numberOfBricks; i++)
    {
        int x = Random.Range(0, countX);
        int y = Random.Range(0, countY);
        int point = Random.Range(1, 5);
        AddBrick(x, y, point);
    }
}

void AddBrick(int x, int y, int point)
{
    int initialX = x, initialY = y;
    brick.GetComponent<BrickScript>().point = point;
    while (x >= 0 && bricks[x, y] != null)
    {
        x--;
    }
    if (x < 0)
    {
        x = initialX;
        while (x < countX && bricks[x, y])
        {
            x++;
        }
        if (x >= countX)
        {
            x = initialX;
            while (y >= 0 && bricks[x, y] != null)
            {
                y--;
            }
            if (y < 0)
            {
                y = initialY;
                while (y < countY && bricks[x, y] != null)
                {
                    y++;
                }
            }
        }
    }
    if (x >= 0 && x < countX && y >= 0 && y < countY)
    {
        bricks[x, y] = Instantiate(brick, new Vector2((x + startX) * 2, y + startY), Quaternion.identity).gameObject;
    }
}
```

The project source code can be found [here](https://github.com/dhiraj-salian/brick-breaker).