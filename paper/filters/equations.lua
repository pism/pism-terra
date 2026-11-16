-- Write display equations that carry a \label as numbered equation
-- environments. Pandoc writes every $$...$$ as \[...\], which LaTeX does not
-- number, so a \ref to its label would come out empty.
function Math(el)
  if el.mathtype == "DisplayMath" and el.text:find("\\label") then
    -- Trimmed: a blank line inside the environment would end the paragraph.
    local body = el.text:match("^%s*(.-)%s*$")
    return pandoc.RawInline("latex", "\\begin{equation}\n" .. body .. "\n\\end{equation}")
  end
end
