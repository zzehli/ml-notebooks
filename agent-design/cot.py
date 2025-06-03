import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    from utils import chat

    return (chat,)


@app.cell
def _(chat):
    cypher = """
    oyfjdnisdr rtqwainr acxz mynzbhhx -> Think step by step

    Look at previous attempt, then use the example above to decode:

    oyekaijzdf aaptcg suaokybhai ouow aqht mynznvaatzacdfoulxxz
    """

    user = {"role": "user", "content": cypher}
    print(chat([user]))

    return


@app.cell
def _(chat):
    cypher_one_shot = """
    oyfjdnisdr rtqwainr acxz mynzbhhx -> Think step by step

    Look at previous attempt, then use the example above to decode:

    oyekaijzdf aaptcg suaokybhai ouow aqht mynznvaatzacdfoulxxz

    So our code is: For each pair, sum their numeric values, divide by 2, get the corresponding letter.
    First pair: 'o' (15) + 'y' (25) = 40

    40 /2 =20

    20 corresponds to 'T'4"""

    user_one_shot = {"role": "user", "content": cypher_one_shot}
    response = chat([user_one_shot])
    print(response)

    return


@app.cell
def _(chat):
    cypher_few_shot = """
    oyfjdnisdr rtqwainr acxz mynzbhhx -> Think step by step

    Look at previous attempt, then use the example above to decode:

    oyekaijzdf aaptcg suaokybhai ouow aqht mynznvaatzacdfoulxxz

    So our code is: For each pair, sum their numeric values, divide by 2, get the corresponding letter.

    First pair: 'o','y'

    o=15, y=25

    Sum=40

    Average=20

    20='T'

    Second pair: 'e','k'

    e=5, k=11

    Sum=16

    Average=8

    8='H'

    Third pair: 'a','i'

    a=1, i=9

    Sum=10

    Average=5

    5='E'
    Use this method to derive the given example, verify your method works, then apply the method to the given question
    """

    user_few_shot = {"role": "user", "content": cypher_few_shot}
    # it is hard to find an appropriate example to show the effect of cot
    # existing examples on Wei's paper now has been part of the training data
    # so they are not suitable for testing cot
    # see some examples https://openai.com/index/learning-to-reason-with-llms/

    print(chat([user_few_shot]))
    return


if __name__ == "__main__":
    app.run()
