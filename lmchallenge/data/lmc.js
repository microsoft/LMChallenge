// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

// *** A Javascript visualizer for LMC logs ***


// Helpers

function to_fixed(x, n) {
    return x === null ? "null" : x.toFixed(n);
}

function percent(x) {
    return to_fixed(100 * x, 1) + " %";
}

// Return the log data, grouped by user's messages
//   data -- a list of events for tokens
//   returns -- a list of list of events for each message
function data_by_line(data) {
    var lines = [];
    var user = null;
    var message = NaN;
    var line = null;
    for (var i = 0; i < data.length; ++i) {
        var d = data[i];
        if (d.user === user && d.message === message) {
            line.push(d);
        } else {
            line = [d];
            lines.push(line);
            user = d.user;
            message = d.message;
        }
    }
    return lines;
}

// Computes summary statistics from the results set, as ratios
function summary_stats(results) {
    // Set up aggregators
    var stats = {"total": 0, "filter_included": 0};
    if (results[0].verbatim) {
        stats.wr = {
            "inaccurate_incorrect": 0,
            "inaccurate_correct": 0,
            "accurate_incorrect": 0,
            "accurate_correct": 0
        };
    }
    if (results[0].logp !== undefined) {
        stats.entropy = {
            "hit": 0,
            "entropy": 0
        };
    }

    // Compute aggregates
    results.forEach(function (d) {
        stats.total += 1;
        if (d.select) {
            stats.filter_included += 1;
            if (stats.wr) {
                var before = d.verbatim === d.target;
                var after = d.results[0][0] === d.target;
                stats.wr.inaccurate_incorrect += !before && !after;
                stats.wr.inaccurate_correct += !before && after;
                stats.wr.accurate_incorrect += before && !after;
                stats.wr.accurate_correct += before && after;
            }
            if (stats.entropy) {
                stats.entropy.hit += (d.logp !== null);
                if (d.logp !== null) {
                    stats.entropy.entropy -= d.logp;
                }
            }
        }
    });

    // "Sum up" stats
    if (stats.wr) {
        stats.wr.inaccurate_incorrect /= stats.filter_included;
        stats.wr.inaccurate_correct /= stats.filter_included;
        stats.wr.accurate_incorrect /= stats.filter_included;
        stats.wr.accurate_correct /= stats.filter_included;
    }
    if (stats.entropy) {
        stats.entropy.entropy /= stats.entropy.hit;
        stats.entropy.hit /= stats.total;
    }
    stats.filter_included /= stats.total;

    return stats;
}


// Rendering

function render_summary(stats) {
    $(".results").empty();
    if (stats.wr) {
        $(".results").append($("<table>").addClass("table").addClass("table-bordered")
            .append($("<tr>")
                    .append($("<td>").addClass("info").text("Filter " + percent(stats.filter_included)))
                    .append("<td>Incorrect</td>")
                    .append("<td>Correct</td>"))
            .append($("<tr>")
                    .append("<td>Inaccurate</td>")
                    .append($("<td>").addClass("warning").text(percent(stats.wr.inaccurate_incorrect)))
                    .append($("<td>").addClass("success").text(percent(stats.wr.inaccurate_correct))))
            .append($("<tr>")
                    .append("<td>Accurate</td>")
                    .append($("<td>").addClass("danger").text(percent(stats.wr.accurate_incorrect)))
                    .append($("<td>").addClass("active").text(percent(stats.wr.accurate_correct))))
        );
    }
    if (stats.entropy) {
        $(".results").append($("<table>").addClass("table").addClass("table-bordered")
            .append($("<tr>")
                    .append("<th>Filter</th>")
                    .append($("<td>").addClass("info").text(percent(stats.filter_included))))
            .append($("<tr>")
                    .append("<th>Hit (after filter)</th>")
                    .append($("<td>").addClass("warning").text(percent(stats.entropy.hit))))
            .append($("<tr>")
                    .append("<th>Entropy</th>")
                    .append($("<td>").addClass("success").text(to_fixed(stats.entropy.entropy, 1))))
        );
    }
}

function set_side_by_side(side_by_side) {
    if (side_by_side) {
        $(".side-by-side").show();
    } else {
        $(".side-by-side").hide();
    }
    $(".word").tooltip(side_by_side ? "disable" : "enable");
}

function render_detail(datum) {
    var detail = $("<div>");

    detail.append($("<p>")
                  .append($("<b>").text("Target: "))
                  .append(datum.target)
                  .append("<br/>")
                  .append($("<b>").text("Corrupted: "))
                  .append(datum.verbatim));

    var table = $("<table>")
        .addClass("table").addClass("table-hover").addClass("table-bordered").addClass("results-table")
        .append("<tr><th>Rank:</th><th>Result:</th><th>Score:</th><th>Error score:</th><th>LM score:</th></tr>");

    for (var i = 0; i < datum.results.length; ++i) {
        var d = datum.results[i];
        var rank = i + 1;
        var entry = $("<tr>").append($("<td>").text(rank))
            .append($("<td>").text(d[0]))
            .append($("<td>").text(to_fixed(d[3], 2)))  // score
            .append($("<td>").text(to_fixed(d[1], 2)))  // error score
            .append($("<td>").text(to_fixed(d[2], 2))); // lm score
        if (d[0] === datum.target) {
            entry.addClass("target-row");
        } else if (d[0] === datum.verbatim) {
            entry.addClass("verbatim-row");
        }
        table.append(entry);
    }
    detail.append(table);

    $(".detail").empty().append(detail);
}

function render_pretty(data) {
    var root = d3.select(".pretty");

    var rows = root.selectAll("p")
        .data(data_by_line(data))
        .enter()
        .append("p")
        .classed("line", true);

    var cells = rows.selectAll("div")
        .data(function (d) { return d; })
        .enter()
        .append("div")
        .classed("word", true);

    // Basic content - the targets themselves
    cells.append("div")
        .text(function (d) { return d.target; });

    // Filter out unselected cells
    cells.classed("filtered", function(d) {
        return !d.select;
    });

    // Only return selected cells
    return cells.filter(function (d) { return d.select; });
}

function render_wr_pretty(data) {
    var cells = render_pretty(data);

    // On click - details
    cells.style("cursor", "pointer")
        .on("click", function (d) {
            $(".word.selected").removeClass("selected");
            $(this).addClass("selected");
            render_detail(d);
            d3.event.stopPropagation();
        });

    var changed_cells = cells.filter(function (d) {
        var r0 = d.results[0][0];
        return d.target !== d.verbatim || d.verbatim !== r0;
    });

    // Tooltips
    changed_cells.attr("data-toggle", "tooltip")
        .attr("data-html", true)
        .attr("data-placement", "bottom")
        .attr("title", function (d) {
            return "<div class=\"tip\"><b>Target: </b>" + d.target +
                "<br/><b>Corrupted: </b>" + d.verbatim +
                "<br/><b>Prediction: </b>" + d.results[0][0] +
                "</div>";
        });

    // Exapandable "side-by-side" content
    changed_cells.append("div")
        .classed("side-by-side", true)
        .text(function (d) { return d.verbatim; });
    changed_cells.append("div")
        .classed("side-by-side", true)
        .text(function (d) { return d.results[0][0]; });
    set_side_by_side($(".wr-side-by-side")[0].checked);

    // Colours
    changed_cells.style("color", function (d) {
        var r0 = d.results[0][0];
        if (d.target !== d.verbatim && d.target === r0) {
            return "#00a030"; // corrected
        } else if (d.target !== d.verbatim && d.target !== r0) {
            return "#ff8000"; // uncorrected
        } else if (d.target === d.verbatim && d.target !== r0) {
            return "#ff0000"; // miscorrected
        } else {
            console.warn("unexpected case - cell should not be selected");
            return "#000000";
        }
    });
}

function set_entropy_min(minLogp) {
    d3.selectAll(".entropy-hit")
        .style("color", function (d) {
            var maxLogp = 0;
            var x = Math.max(0, Math.min(1, (d.logp - minLogp) / (maxLogp - minLogp)));
            return d3.hsl(120 * x, 1, 0.4);
        });
}

function render_entropy_pretty(data) {
    var cells = render_pretty(data);

    // Tooltips
    cells.attr("data-toggle", "tooltip")
        .attr("data-placement", "bottom")
        .attr("title", function (d) {
            return to_fixed(d.logp, 3);
        });

    // Colours
    cells.classed("entropy-miss", function (d) { return d.logp === null; });
    cells.classed("entropy-hit", function (d) { return d.logp !== null; });
    set_entropy_min(parseFloat($(".entropy-min").val()));
}

// Toplevel setup functions

function setup_wr(results, model) {
    $(".wr-only").show();
    $(".entropy-only").hide();
    render_summary(summary_stats(results));
    render_wr_pretty(results);
    $(".wr-settings").text(model);
}

function setup_entropy(results) {
    $(".entropy-only").show();
    $(".wr-only").hide();
    render_summary(summary_stats(results));
    render_entropy_pretty(results);
}

// A little setup on page load

$(function() {
    $(".wr-side-by-side").change(function() {
        set_side_by_side(this.checked);
    });

    // keyup|mouseup as change() doesn't fire as reliably for number input
    $(".entropy-min").bind("keyup mouseup", function() {
        set_entropy_min(this.value);
    });

    $(".pretty").click(function () {
        $(".word.selected").removeClass("selected");
        $(".detail").empty();
    });
});
