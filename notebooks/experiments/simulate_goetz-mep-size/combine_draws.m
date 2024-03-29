function combine_draws(d_base)
d_base = string(d_base);
d_cell = dir(fullfile(d_base, 'draws'));

vec_p_mat = string(d_base) + filesep + 'draws' + filesep + string({d_cell.name});
vec_p_mat = sort(vec_p_mat(not([d_cell.isdir])));

for ix_vec_p_mat = 1:length(vec_p_mat)
    p = load(vec_p_mat(ix_vec_p_mat));

    assert(all(p.parameters(1).ix_draw == [p.parameters.ix_draw]), 'Expecting saving to happen draw-by-draw');

    if not(p.parameters(1).ix_draw == ix_vec_p_mat)
        fprintf('Expecting the index of this loop to match the draw index - stopping combination early.\n')
        break;
    end

    if ix_vec_p_mat == 1
        int_mat_local = p.int_mat;
        parameters = p.parameters;
        rc_mat_local = p.rc_mat;
        v = p.v;
    else
        case_error = not([ ...
            all(v.n_participant == p.v.n_participant, 'all'), ...
            all(v.stimamp_intensity == p.v.stimamp_intensity, 'all'), ...
            all(v.n_reps_search == p.v.n_reps_search, 'all'), ...
            all(v.n_reps_minmax == p.v.n_reps_minmax, 'all'), ...
            all(v.stimamp_saturation == p.v.stimamp_saturation, 'all'), ...
            all(v.stimamp_intensity == p.v.stimamp_intensity, 'all'), ...
            ]);
        if any(case_error)
            % save as far as you got
            save(fullfile(d_base, sprintf('combined_draws.mat')), 'parameters', 'rc_mat_local', 'int_mat_local', 'v');
            error('Mixing draw types!');
        end

        int_mat_local = [int_mat_local; p.int_mat];
        parameters = [parameters; p.parameters];
        rc_mat_local = [rc_mat_local; p.rc_mat];

    end
end

save(fullfile(d_base, sprintf('combined_draws.mat')), 'parameters', 'rc_mat_local', 'int_mat_local', 'v');
end