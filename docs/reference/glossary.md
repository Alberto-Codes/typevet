# Glossary

Kind: reference. This page gives the one meaning of each term that the typevet
pages use.

Every typevet page uses these words with these meanings only. The entries are
in alphabetical order.

## Terms

**acceptance review.** An independent check of a builder's diff against the
accepted issue contract. The reviewer changes no file.

**builder.** The worker role that implements one accepted contract inside named
paths and returns evidence. The builder never commits.

**harness.** The program that runs a worker model and gives it tools. typevet
uses three harnesses: pi, the Claude Code Agent tool and the Cursor CLI.

**pi-fit.** An issue label for mechanical, machine-checkable work that any
verified worker harness can take.

**reviewer.** The role that exercises the defining behaviour and reports
findings. A model name alone does not make a review independent.

**specifier.** The role that writes a definition of ready and done for one
issue. The specifier never edits code.

**supervisor.** The session that selects work, settles decisions, owns
acceptance and commits. The supervisor harness is independent of the worker
harness. Cursor, Claude Code, Codex, Copilot and pi may each supervise.

**worker trailer.** A `Generated-By` or `Specified-By` commit trailer that
records which worker model produced code or a specification. It is evaluation
evidence, not authorship. Never use `Co-Authored-By` for a model.

**TypeLLM.** The type-safe generation stack at
[TypeLLM/TypeLLM](https://github.com/TypeLLM/TypeLLM). typevet evaluates and
hardens work around it.

**worker.** A bounded session that follows a brief, edits only allowed paths
and returns evidence. A worker never commits unless the brief assigns that
action.
