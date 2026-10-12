import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from Bio.Seq import Seq
from io import StringIO

st.set_page_config(
    page_title="Gene2Protein",
    page_icon="🧬",
    layout="wide"
)

STOP_CODONS = {"TAA", "TAG", "TGA"}
VALID_BASES = set("ATGC")


# -------------------- HELPER FUNCTIONS --------------------

def normalize_dna(sequence):
    return "".join(sequence.split()).upper()


def valid_dna(sequence):
    return bool(sequence) and set(sequence).issubset(VALID_BASES)


def gc_content(dna):
    if not dna:
        return 0.0
    return (dna.count("G") + dna.count("C")) / len(dna) * 100


def dna_to_mrna(dna):
    # DNA coding strand -> mRNA
    return dna.replace("T", "U")


def translate_dna(dna):
    # Translate complete codons from the first base
    usable_length = len(dna) - len(dna) % 3
    if usable_length == 0:
        return ""
    return str(Seq(dna[:usable_length]).translate())


def codon_list(dna):
    usable_length = len(dna) - len(dna) % 3
    return [
        dna[i:i + 3]
        for i in range(0, usable_length, 3)
    ]


def classify_substitution(original_dna, mutated_dna, position):
    """Classify a single-base substitution at a 1-based position."""
    start = ((position - 1) // 3) * 3
    old_codon = original_dna[start:start + 3]
    new_codon = mutated_dna[start:start + 3]

    if len(old_codon) != 3 or len(new_codon) != 3:
        return {
            "classification": "Unclassified",
            "old_codon": old_codon,
            "new_codon": new_codon,
            "old_aa": "",
            "new_aa": "",
            "explanation": "The mutation affects an incomplete codon."
        }

    old_aa = str(Seq(old_codon).translate())
    new_aa = str(Seq(new_codon).translate())

    if old_aa == new_aa:
        category = "Synonymous (silent)"
        explanation = "The amino acid remains unchanged."
    elif old_aa != "*" and new_aa == "*":
        category = "Nonsense (stop gained)"
        explanation = "A stop codon has been created."
    elif old_aa == "*" and new_aa != "*":
        category = "Stop-loss"
        explanation = "A stop codon has been removed."
    else:
        category = "Missense"
        explanation = "The encoded amino acid has changed."

    return {
        "classification": category,
        "old_codon": old_codon,
        "new_codon": new_codon,
        "old_aa": old_aa,
        "new_aa": new_aa,
        "explanation": explanation
    }


def compare_codons(original_dna, mutated_dna):
    old_codons = codon_list(original_dna)
    new_codons = codon_list(mutated_dna)
    rows = []

    for i in range(max(len(old_codons), len(new_codons))):
        old = old_codons[i] if i < len(old_codons) else "—"
        new = new_codons[i] if i < len(new_codons) else "—"

        old_aa = str(Seq(old).translate()) if old != "—" else "—"
        new_aa = str(Seq(new).translate()) if new != "—" else "—"

        rows.append({
            "Codon Number": i + 1,
            "Original Codon": old,
            "Mutated Codon": new,
            "Original Amino Acid": old_aa,
            "Mutated Amino Acid": new_aa,
            "Changed?": "Yes" if old_aa != new_aa else "No"
        })

    return rows


def six_frame_translation(dna):
    """Translate three forward and three reverse-complement frames."""
    reverse_dna = str(Seq(dna).reverse_complement())
    results = []

    for strand, sequence in [
        ("Forward (+)", dna),
        ("Reverse (-)", reverse_dna)
    ]:
        for frame in range(3):
            frame_dna = sequence[frame:]
            protein = translate_dna(frame_dna)

            results.append({
                "Reading Frame": f"{strand}, Frame {frame + 1}",
                "DNA Sequence": frame_dna,
                "Protein Sequence": protein
            })

    return results


def detect_orfs(dna):
    """
    Find complete ORFs starting with ATG and ending at the first
    in-frame stop codon, across all six reading frames.
    """
    reverse_dna = str(Seq(dna).reverse_complement())
    sequence_length = len(dna)
    results = []

    for strand, sequence in [
        ("Forward (+)", dna),
        ("Reverse (-)", reverse_dna)
    ]:
        for frame in range(3):
            frame_sequence = sequence[frame:]
            usable_length = len(frame_sequence) - len(frame_sequence) % 3
            frame_sequence = frame_sequence[:usable_length]

            codons = [
                frame_sequence[i:i + 3]
                for i in range(0, len(frame_sequence), 3)
            ]

            for start_index, codon in enumerate(codons):
                if codon != "ATG":
                    continue

                for stop_index in range(start_index + 1, len(codons)):
                    if codons[stop_index] in STOP_CODONS:
                        start = frame + start_index * 3
                        end = frame + (stop_index + 1) * 3

                        orf_dna = sequence[start:end]
                        protein = str(Seq(orf_dna).translate())

                        if strand == "Forward (+)":
                            original_start = start + 1
                            original_end = end
                        else:
                            original_start = sequence_length - end + 1
                            original_end = sequence_length - start

                        results.append({
                            "Strand": strand,
                            "Frame": frame + 1,
                            "Original DNA Range (1-based)": (
                                f"{original_start}-{original_end}"
                            ),
                            "Length (bp)": len(orf_dna),
                            "Start Codon": "ATG",
                            "Stop Codon": codons[stop_index],
                            "Protein Sequence": protein
                        })

                        # First in-frame stop for this start codon
                        break

    return results



# -------------------- VISUALIZATION FUNCTIONS --------------------

BASE_COLORS = {
    "A": "#3498DB",
    "T": "#E67E22",
    "G": "#2ECC71",
    "C": "#9B59B6"
}


def dna_visualization(original, mutated):
    """Show original and mutated DNA with changed positions highlighted."""
    fig = go.Figure()

    sequences = [
        ("Original DNA", original),
        ("Mutated DNA", mutated)
    ]

    for row, (label, sequence) in enumerate(sequences):
        colors = []

        for i, base in enumerate(sequence):
            changed = (
                row == 1
                and (
                    i >= len(original)
                    or original[i] != base
                )
            )

            colors.append("#E74C3C" if changed else BASE_COLORS[base])

        fig.add_trace(go.Scatter(
            x=list(range(1, len(sequence) + 1)),
            y=[label] * len(sequence),
            mode="markers+text",
            text=list(sequence),
            textposition="middle center",
            marker=dict(
                size=30,
                color=colors,
                line=dict(color="white", width=1)
            ),
            hovertemplate=(
                "Position: %{x}<br>"
                "Nucleotide: %{text}<extra></extra>"
            ),
            name=label
        ))

    fig.update_layout(
        title="DNA Mutation Map",
        xaxis_title="Nucleotide Position",
        yaxis_title="Sequence",
        template="plotly_white",
        height=260,
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=40)
    )

    return fig


def protein_visualization(original_protein, mutated_protein):
    """Compare original and mutated amino acid sequences."""
    fig = go.Figure()

    sequences = [
        ("Original Protein", original_protein),
        ("Mutated Protein", mutated_protein)
    ]

    for row, (label, sequence) in enumerate(sequences):
        colors = []

        for i, amino_acid in enumerate(sequence):
            changed = (
                row == 1
                and (
                    i >= len(original_protein)
                    or original_protein[i] != amino_acid
                )
            )

            if amino_acid == "*":
                colors.append("#34495E")
            elif changed:
                colors.append("#E74C3C")
            else:
                colors.append("#16A085")

        fig.add_trace(go.Scatter(
            x=list(range(1, len(sequence) + 1)),
            y=[label] * len(sequence),
            mode="markers+text",
            text=list(sequence),
            textposition="middle center",
            marker=dict(
                size=30,
                color=colors,
                line=dict(color="white", width=1)
            ),
            hovertemplate=(
                "Amino acid position: %{x}<br>"
                "Symbol: %{text}<extra></extra>"
            ),
            name=label
        ))

    fig.update_layout(
        title="Protein Sequence Comparison",
        xaxis_title="Amino Acid Position",
        yaxis_title="Sequence",
        template="plotly_white",
        height=260,
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=40)
    )

    return fig


def codon_visualization(dna):
    """Show codons and their translated amino acids."""
    codons = codon_list(dna)

    if not codons:
        return None

    amino_acids = [
        str(Seq(codon).translate())
        for codon in codons
    ]

    fig = go.Figure()

    for i, (codon, amino_acid) in enumerate(
        zip(codons, amino_acids)
    ):
        color = "#34495E" if amino_acid == "*" else "#3498DB"

        fig.add_trace(go.Bar(
            x=[f"Codon {i + 1}"],
            y=[1],
            text=[f"{codon}<br>↓<br>{amino_acid}"],
            textposition="inside",
            marker_color=color,
            name=f"{codon} → {amino_acid}",
            hovertemplate=(
                f"Codon: {codon}<br>"
                f"Amino acid: {amino_acid}<extra></extra>"
            ),
            showlegend=False
        ))

    fig.update_layout(
        title="Codon → Amino Acid Translation",
        xaxis_title="Codon",
        yaxis=dict(visible=False, range=[0, 1.3]),
        template="plotly_white",
        height=260,
        margin=dict(l=20, r=20, t=60, b=40)
    )

    return fig


def orf_map_visualization(dna, orf_results):
    """Visualize detected ORFs against the original DNA coordinates."""
    fig = go.Figure()
    dna_length = len(dna)

    # Original DNA sequence baseline
    fig.add_trace(go.Scatter(
        x=[1, dna_length],
        y=[0, 0],
        mode="lines",
        line=dict(color="#64748b", width=4),
        hoverinfo="skip",
        showlegend=False
    ))

    # Display each nucleotide above the baseline
    fig.add_trace(go.Scatter(
        x=list(range(1, dna_length + 1)),
        y=[0.12] * dna_length,
        mode="text",
        text=list(dna),
        textposition="top center",
        textfont=dict(size=12),
        hovertemplate="Position: %{x}<br>Base: %{text}<extra></extra>",
        showlegend=False
    ))

    for i, orf in enumerate(orf_results):
        range_text = orf["Original DNA Range (1-based)"]
        start, end = map(int, range_text.split("-"))
        strand = orf["Strand"]
        y = -(i + 1)

        is_reverse = "Reverse" in strand

        # Reverse-strand ORFs run in the opposite direction
        if is_reverse:
            start_codon_mid = end - 1
            stop_codon_mid = start + 1
            strand_label = "Reverse (−)"
        else:
            start_codon_mid = start + 1
            stop_codon_mid = end - 1
            strand_label = "Forward (+)"

        # ORF region
        fig.add_shape(
            type="rect",
            x0=start - 0.5,
            x1=end + 0.5,
            y0=y - 0.22,
            y1=y + 0.22,
            fillcolor="#3498DB" if not is_reverse else "#9B59B6",
            opacity=0.35,
            line=dict(
                color="#3498DB" if not is_reverse else "#9B59B6",
                width=2
            )
        )

        # Start codon marker
        fig.add_trace(go.Scatter(
            x=[start_codon_mid],
            y=[y],
            mode="markers+text",
            text=["ATG"],
            textposition="top center",
            marker=dict(color="#27AE60", size=12, symbol="triangle-up"),
            name="Start codon",
            showlegend=(i == 0),
            hovertemplate=(
                f"{strand_label} ORF<br>"
                f"Start codon: ATG<br>"
                f"Original DNA position: {start_codon_mid}<extra></extra>"
            )
        ))

        # Stop codon marker
        fig.add_trace(go.Scatter(
            x=[stop_codon_mid],
            y=[y],
            mode="markers+text",
            text=["STOP"],
            textposition="bottom center",
            marker=dict(color="#E74C3C", size=11, symbol="x"),
            name="Stop codon",
            showlegend=(i == 0),
            hovertemplate=(
                f"{strand_label} ORF<br>"
                f"Stop codon<br>"
                f"Original DNA position: {stop_codon_mid}<extra></extra>"
            )
        ))

        fig.add_annotation(
            x=end if not is_reverse else start,
            y=y + 0.35,
            text=f"ORF {i + 1} | {strand_label} | {start}-{end}",
            showarrow=False,
            font=dict(size=10)
        )

    fig.update_layout(
        title="Open Reading Frame (ORF) Map",
        xaxis_title="Position on Original DNA (1-based)",
        yaxis=dict(
            visible=False,
            range=[-max(len(orf_results), 1) - 0.8, 0.8]
        ),
        xaxis=dict(
            range=[0.5, dna_length + 0.5],
            dtick=1
        ),
        template="plotly_white",
        height=max(350, 130 + len(orf_results) * 75),
        margin=dict(l=20, r=20, t=65, b=50)
    )

    return fig


def mutation_impact_summary(original_dna, mutated_dna):
    original_protein = translate_dna(original_dna)
    mutated_protein = translate_dna(mutated_dna)

    original_length = len(original_dna)
    mutated_length = len(mutated_dna)
    length_change = mutated_length - original_length

    # Compare proteins position-wise
    max_length = max(len(original_protein), len(mutated_protein))
    changed_amino_acids = []

    for i in range(max_length):
        old_aa = original_protein[i] if i < len(original_protein) else "-"
        new_aa = mutated_protein[i] if i < len(mutated_protein) else "-"

        if old_aa != new_aa:
            changed_amino_acids.append({
                "Position": i + 1,
                "Original": old_aa,
                "Mutated": new_aa
            })

    if length_change != 0:
        if length_change % 3 != 0:
            mutation_effect = "Potential frameshift"
        else:
            mutation_effect = "In-frame insertion/deletion"
    elif original_protein == mutated_protein:
        mutation_effect = "No protein sequence change"
    else:
        mutation_effect = "Protein sequence changed"

    return {
        "Original DNA Length": original_length,
        "Mutated DNA Length": mutated_length,
        "DNA Length Change": length_change,
        "Original Protein": original_protein,
        "Mutated Protein": mutated_protein,
        "Changed Amino Acids": changed_amino_acids,
        "Mutation Effect": mutation_effect
    }




# -------------------- MAIN APPLICATION --------------------

st.title("🧬 Gene2Protein")
st.subheader(
    "DNA Analysis | Mutation Simulator | Effect Classification"
)

st.write(
    "Explore DNA mutations, codons, translated proteins, "
    "reading frames, and potential open reading frames."
)


# 1. ORIGINAL DNA ANALYSIS

st.header("1. Original DNA Analysis")

dna_input = st.text_area(
    "DNA coding strand (5′ → 3′)",
    value="ATGGCTTTTGAATGA",
    height=100,
    help="Enter a sequence containing only A, T, G, and C."
)

dna = normalize_dna(dna_input)

if not valid_dna(dna):
    st.error(
        "Enter a non-empty DNA sequence containing only A, T, G, and C."
    )
    st.stop()

original_protein = translate_dna(dna)
mrna = dna_to_mrna(dna)
codons = codon_list(dna)

col1, col2, col3 = st.columns(3)
col1.metric("DNA Length", f"{len(dna)} bp")
col2.metric("GC Content", f"{gc_content(dna):.2f}%")
col3.metric("Complete Codons", len(codons))

with st.expander("View DNA, mRNA, codons and protein", expanded=True):
    st.text(f"DNA:     {dna}")
    st.text(f"mRNA:    {mrna}")
    st.text(f"Codons:  {' | '.join(codons)}")
    st.text(f"Protein: {original_protein}")

    st.caption(
        "The * symbol represents a stop codon. "
        "Incomplete trailing bases are not translated."
    )


# 2. MUTATION SIMULATOR

st.header("2. Mutation Simulator")

mutation_type = st.selectbox(
    "Mutation type",
    ["Substitution", "Insertion", "Deletion"]
)

if mutation_type == "Insertion":
    position = st.number_input(
        "Insert after position (0 = before the first base)",
        min_value=0,
        max_value=len(dna),
        value=min(3, len(dna)),
        step=1
    )
else:
    position = st.number_input(
        "Position (1-based)",
        min_value=1,
        max_value=len(dna),
        value=min(4, len(dna)),
        step=1
    )

new_base = None

if mutation_type in ["Substitution", "Insertion"]:
    new_base = st.selectbox(
        "New nucleotide",
        ["A", "T", "G", "C"]
    )

if mutation_type == "Substitution":
    mutated_dna = (
        dna[:int(position) - 1]
        + new_base
        + dna[int(position):]
    )

    st.caption(
        f"Original nucleotide at position {int(position)}: "
        f"{dna[int(position) - 1]}"
    )

elif mutation_type == "Insertion":
    mutated_dna = (
        dna[:int(position)]
        + new_base
        + dna[int(position):]
    )

    st.caption(f"Inserted {new_base} after position {int(position)}.")

else:
    mutated_dna = (
        dna[:int(position) - 1]
        + dna[int(position):]
    )

    st.caption(
        f"Deleted nucleotide {dna[int(position) - 1]} "
        f"at position {int(position)}."
    )

st.write("Mutated DNA preview:")
st.code(mutated_dna, language=None)

mutated_protein = translate_dna(mutated_dna)
mutated_mrna = dna_to_mrna(mutated_dna)


# 3. MUTATION ANALYSIS

st.header("3. Mutation Analysis")

original_col, mutated_col = st.columns(2)

with original_col:
    st.markdown("**Original**")
    st.code(dna, language=None)
    st.write("Protein:")
    st.code(original_protein or "(no complete codons)", language=None)

with mutated_col:
    st.markdown("**Mutated**")
    st.code(mutated_dna, language=None)
    st.write("Protein:")
    st.code(mutated_protein or "(no complete codons)", language=None)

st.write(f"Original length: **{len(dna)} bp**")
st.write(f"Mutated length: **{len(mutated_dna)} bp**")
st.write(f"Length change: **{len(mutated_dna) - len(dna):+d} bp**")


# -------------------- MUTATION IMPACT DASHBOARD --------------------
st.header("📊 Mutation Impact Dashboard")

if mutated_dna != dna:
    impact = mutation_impact_summary(dna, mutated_dna)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Original DNA",
            f"{impact['Original DNA Length']} bp"
        )

    with col2:
        st.metric(
            "Mutated DNA",
            f"{impact['Mutated DNA Length']} bp",
            delta=f"{impact['DNA Length Change']:+} bp"
        )

    with col3:
        st.metric(
            "Changed Amino Acids",
            len(impact["Changed Amino Acids"])
        )

    st.subheader("Protein Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original Protein**")
        st.code(impact["Original Protein"] or "(empty)")

    with col2:
        st.markdown("**Mutated Protein**")
        st.code(impact["Mutated Protein"] or "(empty)")

    st.info(f"**Observed effect:** {impact['Mutation Effect']}")

    if impact["Changed Amino Acids"]:
        st.subheader("Changed Amino Acid Positions")
        st.dataframe(
            impact["Changed Amino Acids"],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.success("No amino acid differences detected.")

else:
    st.info("Select a mutation in the Mutation Simulator to view its impact.")




# -------------------- VISUALIZATION DASHBOARD --------------------

st.header("📊  Interactive Visualization Dashboard")

st.subheader("1. DNA Sequence and Mutation")

st.caption(
    "Blue = A, Orange = T, Green = G, Purple = C. "
    "Red indicates a changed nucleotide in the mutated sequence."
)

st.plotly_chart(
    dna_visualization(dna, mutated_dna),
    use_container_width=True
)

st.subheader("2. DNA Codon Translation")

codon_fig = codon_visualization(dna)

if codon_fig is not None:
    st.plotly_chart(
        codon_fig,
        use_container_width=True
    )

st.subheader("3. Protein Sequence Comparison")

st.caption(
    "Red indicates a changed amino acid. "
    "The dark marker represents a stop codon."
)

st.plotly_chart(
    protein_visualization(
        original_protein,
        mutated_protein
    ),
    use_container_width=True
)

st.subheader("4. Six Reading Frames Overview")

frame_results = six_frame_translation(dna)

frame_cols = st.columns(3)

for index, frame in enumerate(frame_results):
    with frame_cols[index % 3]:
        st.markdown(f"**{frame['Reading Frame']}**")
        st.code(
            frame["Protein Sequence"] or "(no complete codons)",
            language=None
        )


# 4. CODON COMPARISON

st.header("4. Codon Comparison")

comparison_rows = compare_codons(dna, mutated_dna)

st.dataframe(
    pd.DataFrame(comparison_rows),
    use_container_width=True,
    hide_index=True
)


# 5. MUTATION EFFECT

st.header("5. Mutation Effect")

effect = None

if mutation_type == "Substitution":
    effect = classify_substitution(
        dna,
        mutated_dna,
        int(position)
    )

    st.markdown(
        f"**Predicted Sequence Effect: {effect['classification']}**"
    )

    st.write(
        f"Codon change: `{effect['old_codon']} → {effect['new_codon']}`"
    )

    st.write(
        f"Amino acid: `{effect['old_aa']} → {effect['new_aa']}`"
    )

    st.info(effect["explanation"])

else:
    length_change = len(mutated_dna) - len(dna)

    if length_change % 3 != 0:
        st.warning(
            "Potential frameshift: the number of inserted or deleted "
            "bases is not divisible by 3. Downstream codons may change."
        )
    else:
        st.info(
            "The length change is divisible by 3, so it does not shift "
            "the downstream reading frame in this simple model."
        )

    st.caption(
        "Insertions and deletions are summarized by sequence and length. "
        "They are not assigned a single missense or nonsense label."
    )

st.caption(
    "These educational predictions assume the entered sequence is a "
    "coding strand translated from its first nucleotide using the "
    "standard genetic code. They are not clinical interpretations. "
    "Codon comparison after indels is positional, not a full alignment."
)


# 6. SIX READING FRAMES

st.header("6. Six Reading Frame Translation")

st.write(
    "Translation in three forward frames and three "
    "reverse-complement frames."
)

frame_results = six_frame_translation(dna)

st.dataframe(
    pd.DataFrame([
        {
            "Reading Frame": item["Reading Frame"],
            "Protein Sequence": item["Protein Sequence"]
        }
        for item in frame_results
    ]),
    use_container_width=True,
    hide_index=True
)

with st.expander("View DNA sequence for each frame"):
    st.dataframe(
        pd.DataFrame(frame_results),
        use_container_width=True,
        hide_index=True
    )


# 7. ORF DETECTION

st.header("7. Open Reading Frame (ORF) Detection")

st.write(
    "Find candidate regions starting with ATG and ending at the first "
    "in-frame stop codon across all six reading frames."
)

orf_results = detect_orfs(dna)

if orf_results:
    st.success(f"Detected {len(orf_results)} complete ORF(s).")

    st.dataframe(
        pd.DataFrame(orf_results),
        use_container_width=True,
        hide_index=True
    )
else:
    st.info(
        "No complete ATG-to-stop ORFs were detected in this sequence."
    )

st.caption(
    "An ORF is a candidate coding region, not proof of a functional gene. "
    "This basic detector uses ATG and the standard genetic code."
)


# -------------------- ORF MAP VISUALIZATION --------------------
st.subheader("🧬 ORF Map Visualization")

orf_results = detect_orfs(dna)

if orf_results:
    st.caption(
        "Green triangle = Start codon | Red X = Stop codon | "
        "Blue region = Forward strand | Purple region = Reverse strand"
    )

    st.plotly_chart(
        orf_map_visualization(dna, orf_results),
        use_container_width=True
    )

    st.success(f"Detected {len(orf_results)} ORF(s) across six reading frames.")

else:
    st.info(
        "No complete ORFs detected. An ORF requires a start codon "
        "(ATG) followed by an in-frame stop codon."
    )


# 8. DOWNLOAD REPORT

st.header("8. Download Report")

report = StringIO()

report.write("Gene2Protein Analysis Report\n")
report.write("============================\n\n")
report.write(f"Original DNA: {dna}\n")
report.write(f"Original mRNA: {mrna}\n")
report.write(f"Original protein: {original_protein}\n")
report.write(f"DNA length: {len(dna)} bp\n")
report.write(f"GC content: {gc_content(dna):.2f}%\n\n")

report.write(f"Mutation type: {mutation_type}\n")
report.write(f"Mutation position: {int(position)}\n")
report.write(f"Mutated DNA: {mutated_dna}\n")
report.write(f"Mutated mRNA: {mutated_mrna}\n")
report.write(f"Mutated protein: {mutated_protein}\n")
report.write(
    f"Length change: {len(mutated_dna) - len(dna):+d} bp\n\n"
)

if effect is not None:
    report.write(
        f"Mutation classification: {effect['classification']}\n"
    )
    report.write(
        f"Codon change: {effect['old_codon']} -> {effect['new_codon']}\n"
    )
    report.write(
        f"Amino acid change: {effect['old_aa']} -> {effect['new_aa']}\n"
    )
    report.write(f"Explanation: {effect['explanation']}\n\n")

report.write("Six Reading Frames\n")
for item in frame_results:
    report.write(
        f"{item['Reading Frame']}: {item['Protein Sequence']}\n"
    )

report.write("\nDetected ORFs\n")

if orf_results:
    for number, item in enumerate(orf_results, start=1):
        report.write(
            f"{number}. Strand: {item['Strand']}; "
            f"Frame: {item['Frame']}; "
            f"Range: {item['Original DNA Range (1-based)']}; "
            f"Length: {item['Length (bp)']} bp; "
            f"Stop: {item['Stop Codon']}; "
            f"Protein: {item['Protein Sequence']}\n"
        )
else:
    report.write("No complete ORFs detected.\n")

st.download_button(
    label="📥 Download Analysis Report",
    data=report.getvalue(),
    file_name="gene2protein_report.txt",
    mime="text/plain"
)

st.success("Analysis complete! Try different DNA sequences and mutations.")
