# Sharpening

This submodule enables the sharpening of the grammars generated by ProgSynth reducing drastically their size.
Since ProgSynth tries to enumerate all programs from a grammar reducing its size is a relevant way to speed up the search, sharpening removes non relevant programs increasing the chance of finding quickly a solution to your task.
There are currently two ways to do sharpening and a script for the PBE specification which tries to find automatically empiricallly such constaints and encode them.

<!-- toc -->
Table of contents:

- [Forbidden Patterns](#forbidden-patterns)
- [Simplifying Rules](#simplifying-rules)
  - [Syntax](#syntax)
  - [An Example](#an-example)
  - [Automatically Generated Equations](#automatically-generated-equations)
  - [Limits](#limits)

<!-- tocstop -->

## Forbidden Patterns

The first way to add constraints is when a DSL is instantiated. The second argument given is ``forbidden_patterns: Dict[Tuple[str, int], Str[str]]``.
A key is a tuple ``(name_of_parent_primitive, arg_no)`` and gives access to the set of all primitives that cannot be directly derived for this specific argument of the ``parent_primitive``.
For example:

```python
forbidden = { 
    ("+", 1) : {"+"}, 
    ("-", 0): {"-", "+"}
}
```

We forbid the second argument of ``+`` from being ``+`` and the first argument of ``-`` from being ``-`` or ``+``.

This mechanism is quite powerful but does not enable to encode all constraints however it has the advantage of having no drawbacks, it can also be done within the other framework.

## Simplifying Rules

More generally, we would like to remove regular tree languages from the grammar which also describes a regular tree language.
In order to do that, we define a syntax that describes a subset of regular tree language but covers the most relevant options for program synthesis.
This is quite straightforward with the following code:

```python
from synth.pruning.constraints import add_dfta_constraints
from synth.syntax.grammars import UCFG


cfg = ... # your regular CFG
my_constraints = [...] # a list of string that express the constraints 
sketch = ... # your sketch or None if there isn't one
ucfg = UCFG.from_DFTA_with_ngrams(
        add_dfta_constraints(cfg, my_constraints, global_constraint, progress=False), 2
    )

# You can continue as usual though since it is an UCFG instead of a CFG
# Det objects need to be replaced by U objects
# e.g.: ProbDetGrammar -> ProbUGrammar, HeapSearch -> UHeapSearch, ...
```

### Syntax

Let us write ``P`` for the set of primitives names and variables. Rules are manipulating a set of names:

```
NSet := f1,...,fk | ^f1,...,fk | _
```

where ``f1,...,fk`` are from ``P``, ``^`` is the complement operator and ``_`` represents any symbol.
Rules have the following form:

```
Rules = (NSet Rules1 . . . Rulesk ) | #[NSet]<=N | #[NSet]>=N
```

where ``k`` and ``N`` are constant integers. The rule ``(f g,h _)`` specifies that for each occurrence of ``f``, its first argument must be either ``g`` or ``h``, and the second argument can be anything.
The rule ``#[Var0]>=1`` specifies that ``Var0`` appears at least once.
Remark that ``#[_]<=10`` says that the whole program has size at most 10.
Rules can be nested, for instance ``f #[g]<=1 #[h]>=1`` : for each occurrence of ``f``, the first argument contains at most one ``g``, and the second argument at least one ``h``.
Rules specify locally which primitives or variables can be combined together.

A sketch has the same syntax as a constraint.
It specifies how the solution program should be, that is the derivations from the root, whereas constraints specify derivation anywhere in the programs.
For instance, the skecth ``(f _ g)`` specifies that the program starts with ``f`` and that the second argument of that particular ``f`` is ``g``.

### An Example

Let us consider the grammar of Boolean formulas over the binary operators ``And``, ``Or`` and unary ``Not``, with Boolean variables ``Var0``, ``Var1``.

```
bool -> And(bool, bool) | Or(bool, bool) | Not(bool) | Var0 | Var1
```

Clearly, this grammar generates a lot of redundant programs. Let us specify some rules in order to enforce that all programs are in conjunctive normal form:

```
Or ¬And ¬And ;
Not ¬{And, Or}
```

The first rule specifies that ``And`` cannot be an argument of ``Or`` and the second one that ``And`` and ``Or`` cannot be arguments of ``Not``. The output of a compilation algorithm using these two rules could be the following grammar:

```
bool1 -> And(bool1, bool1 ) | Or(bool2 , bool2 ) | Not(bool3 ) | Var0 | Var1
bool2 -> Or(bool2 , bool2) | Not(bool3) | Var0 | Var1
bool3 -> Not(bool3) | Var0 | Var1
```

There are still a lot of equivalent programs generated by this grammar. One could consider the following rules further reducing symmetries:

```
And ¬And _ ;
Or ¬{Or, And} ¬And ;
Not ¬{And, Or}
```

It ensures conjunctive normal form but also that both ``And`` and ``Or`` are associated to the right: in particular, the formula ``And(And(φ1 , φ2), φ3)`` is replaced
by ``And(φ1 , And(φ2 , φ3))``.

### Automatically Generated Equations

Since writing rules can be tedious, even more so for large grammars, we propose
an automated process for generating valid equations:

1. We enumerate all programs up to some fixed depth (in practice, 3 or 4);
2. We check for program equivalence amongst all generated programs;
3. For each equivalence class of programs, we choose as representative the small-
est program of that class: the goal is to find rules rejecting all non represen-
tative programs;
4. We enumerate rules and for each check whether they are useful, meaning
reject only (new) non representative programs.
Note that program equivalence may be hard to solve; in practice we evaluate the
programs on a set of inputs that is either sampled or scrapped from a dataset
and declare two programs equivalent if their outputs coincide on them.

ProgSynth provides for some type of specifications such as PBE a script ``dsl_analyser.py`` such a tool.
The script works by either reproducing the distriution from a given dataset or just taking inputs from a given dataset, this enables it to produce inputs to test programs.
It then evaluates programs up to depth 2 of the grammar and builds sets of semantically equivalent programs with respect to this set of inputs.
Some of these programs can then be removed from the grammar.
The script produces two files:

- a python file containing all constraints that were automatically produced;
- a JSON file containing all semantically equivalent classes of programs et depth 2.

### Limits

The design of our syntax was guided by simplicity; although expressive enough for most use cases, it could be extended. 
Indeed, some natural rules cannot be expressed, for instance forbidding the pattern ``(f a b)``, when there is an occurrence of ``f`` where the first argument is ``a`` and the second argument is ``b``.
To put this remark in a wider perspective: we note that all simplifyig rules defined in our syntax induce regular tree languages, but conversely that some regular tree languages, such as ‘trees without the pattern ``(f a b)``’, cannot be defined in our syntax.
However, with some small tweaking one can directly give an deterministic bottom-up tree automaton and use it as if it were a rule.