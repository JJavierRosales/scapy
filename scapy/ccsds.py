# Import utils library
from . import utils

# Declare global variable containing all keys within every cluster in a CDM.
cdm_features = {
    'CCSDS_CDM_VERS':               dict(cluster='header', obligatory=True, dtype='string'), 
    'CREATION_DATE':                dict(cluster='header', obligatory=True, dtype='datetime64[ns]'), 
    'ORIGINATOR':                   dict(cluster='header', obligatory=True, dtype='string'), 
    'MESSAGE_FOR':                  dict(cluster='header', obligatory=False, dtype='string'), 
    'MESSAGE_ID':                   dict(cluster='header', obligatory=True, dtype='string'),

    'TCA':                          dict(cluster='relative_metadata', obligatory=True, dtype='datetime64[ns]'), 
    'MISS_DISTANCE':                dict(cluster='relative_metadata', obligatory=True, dtype='float'), 
    'RELATIVE_SPEED':               dict(cluster='relative_metadata', obligatory=False, dtype='float'), 
    'RELATIVE_POSITION_R':          dict(cluster='relative_metadata', obligatory=False, dtype='float'), 
    'RELATIVE_POSITION_T':          dict(cluster='relative_metadata', obligatory=False, dtype='float'), 
    'RELATIVE_POSITION_N':          dict(cluster='relative_metadata', obligatory=False, dtype='float'), 
    'RELATIVE_VELOCITY_R':          dict(cluster='relative_metadata', obligatory=False, dtype='float'), 
    'RELATIVE_VELOCITY_T':          dict(cluster='relative_metadata', obligatory=False, dtype='float'), 
    'RELATIVE_VELOCITY_N':          dict(cluster='relative_metadata', obligatory=False, dtype='float'), 
    'START_SCREEN_PERIOD':          dict(cluster='relative_metadata', obligatory=False, dtype='datetime64[ns]'), 
    'STOP_SCREEN_PERIOD':           dict(cluster='relative_metadata', obligatory=False, dtype='datetime64[ns]'), 
    'SCREEN_VOLUME_FRAME':          dict(cluster='relative_metadata', obligatory=False, dtype='string'), 
    'SCREEN_VOLUME_SHAPE':          dict(cluster='relative_metadata', obligatory=False, dtype='string'), 
    'SCREEN_VOLUME_X':              dict(cluster='relative_metadata', obligatory=False, dtype='float'), 
    'SCREEN_VOLUME_Y':              dict(cluster='relative_metadata', obligatory=False, dtype='float'), 
    'SCREEN_VOLUME_Z':              dict(cluster='relative_metadata', obligatory=False, dtype='float'), 
    'SCREEN_ENTRY_TIME':            dict(cluster='relative_metadata', obligatory=False, dtype='datetime64[ns]'), 
    'SCREEN_EXIT_TIME':             dict(cluster='relative_metadata', obligatory=False, dtype='datetime64[ns]'), 
    'COLLISION_PROBABILITY':        dict(cluster='relative_metadata', obligatory=False, dtype='float'), 
    'COLLISION_PROBABILITY_METHOD': dict(cluster='relative_metadata', obligatory=False, dtype='string'),
  
    'OBJECT_DESIGNATOR':            dict(cluster='metadata', obligatory=True, dtype='string'), 
    'CATALOG_NAME':                 dict(cluster='metadata', obligatory=True, dtype='string'), 
    'OBJECT_NAME':                  dict(cluster='metadata', obligatory=True, dtype='string'), 
    'INTERNATIONAL_DESIGNATOR':     dict(cluster='metadata', obligatory=True, dtype='string'), 
    'OBJECT_TYPE':                  dict(cluster='metadata', obligatory=False, dtype='category'), 
    'OPERATOR_CONTACT_POSITION':    dict(cluster='metadata', obligatory=False, dtype='string'), 
    'OPERATOR_ORGANIZATION':        dict(cluster='metadata', obligatory=False, dtype='string'), 
    'OPERATOR_PHONE':               dict(cluster='metadata', obligatory=False, dtype='string'), 
    'OPERATOR_EMAIL':               dict(cluster='metadata', obligatory=False, dtype='string'), 
    'EPHEMERIS_NAME':               dict(cluster='metadata', obligatory=True, dtype='string'), 
    'COVARIANCE_METHOD':            dict(cluster='metadata', obligatory=True, dtype='category'), 
    'MANEUVERABLE':                 dict(cluster='metadata', obligatory=True, dtype='category'), 
    'ORBIT_CENTER':                 dict(cluster='metadata', obligatory=False, dtype='string'), 
    'REF_FRAME':                    dict(cluster='metadata', obligatory=True, dtype='category'), 
    'GRAVITY_MODEL':                dict(cluster='metadata', obligatory=False, dtype='string'), 
    'ATMOSPHERIC_MODEL':            dict(cluster='metadata', obligatory=False, dtype='string'), 
    'N_BODY_PERTURBATIONS':         dict(cluster='metadata', obligatory=False, dtype='string'), 
    'SOLAR_RAD_PRESSURE':           dict(cluster='metadata', obligatory=False, dtype='category'), 
    'EARTH_TIDES':                  dict(cluster='metadata', obligatory=False, dtype='category'), 
    'INTRACK_THRUST':               dict(cluster='metadata', obligatory=False, dtype='category'),

    'TIME_LASTOB_START':            dict(cluster='data_od', obligatory=False, dtype='datetime64[ns]'), 
    'TIME_LASTOB_END':              dict(cluster='data_od', obligatory=False, dtype='datetime64[ns]'), 
    'RECOMMENDED_OD_SPAN':          dict(cluster='data_od', obligatory=False, dtype='float'), 
    'ACTUAL_OD_SPAN':               dict(cluster='data_od', obligatory=False, dtype='float'), 
    'OBS_AVAILABLE':                dict(cluster='data_od', obligatory=False, dtype='float'), 
    'OBS_USED':                     dict(cluster='data_od', obligatory=False, dtype='float'), 
    'TRACKS_AVAILABLE':             dict(cluster='data_od', obligatory=False, dtype='float'), 
    'TRACKS_USED':                  dict(cluster='data_od', obligatory=False, dtype='float'), 
    'RESIDUALS_ACCEPTED':           dict(cluster='data_od', obligatory=False, dtype='float'), 
    'WEIGHTED_RMS':                 dict(cluster='data_od', obligatory=False, dtype='float'), 
    'AREA_PC':                      dict(cluster='data_od', obligatory=False, dtype='float'),  
    'AREA_DRG':                     dict(cluster='data_od', obligatory=False, dtype='float'),  
    'AREA_SRP':                     dict(cluster='data_od', obligatory=False, dtype='float'),  
    'MASS':                         dict(cluster='data_od', obligatory=False, dtype='float'),  
    'CD_AREA_OVER_MASS':            dict(cluster='data_od', obligatory=False, dtype='float'),  
    'CR_AREA_OVER_MASS':            dict(cluster='data_od', obligatory=False, dtype='float'),  
    'THRUST_ACCELERATION':          dict(cluster='data_od', obligatory=False, dtype='float'),  
    'SEDR':                         dict(cluster='data_od', obligatory=False, dtype='float'), 

    'X':                            dict(cluster='data_state', obligatory=True, dtype='float'), 
    'Y':                            dict(cluster='data_state', obligatory=True, dtype='float'), 
    'Z':                            dict(cluster='data_state', obligatory=True, dtype='float'), 
    'X_DOT':                        dict(cluster='data_state', obligatory=True, dtype='float'), 
    'Y_DOT':                        dict(cluster='data_state', obligatory=True, dtype='float'), 
    'Z_DOT':                        dict(cluster='data_state', obligatory=True, dtype='float'),

    'CR_R':                         dict(cluster='data_covariance', obligatory=True, dtype='float'), 
    'CT_R':                         dict(cluster='data_covariance', obligatory=True, dtype='float'), 
    'CT_T':                         dict(cluster='data_covariance', obligatory=True, dtype='float'), 
    'CN_R':                         dict(cluster='data_covariance', obligatory=True, dtype='float'), 
    'CN_T':                         dict(cluster='data_covariance', obligatory=True, dtype='float'), 
    'CN_N':                         dict(cluster='data_covariance', obligatory=True, dtype='float'), 
    'CRDOT_R':                      dict(cluster='data_covariance', obligatory=True, dtype='float'), 
    'CRDOT_T':                      dict(cluster='data_covariance', obligatory=True, dtype='float'), 
    'CRDOT_N':                      dict(cluster='data_covariance', obligatory=True, dtype='float'), 
    'CRDOT_RDOT':                   dict(cluster='data_covariance', obligatory=True, dtype='float'), 
    'CTDOT_R':                      dict(cluster='data_covariance', obligatory=True, dtype='float'), 
    'CTDOT_T':                      dict(cluster='data_covariance', obligatory=True, dtype='float'), 
    'CTDOT_N':                      dict(cluster='data_covariance', obligatory=True, dtype='float'), 
    'CTDOT_RDOT':                   dict(cluster='data_covariance', obligatory=True, dtype='float'), 
    'CTDOT_TDOT':                   dict(cluster='data_covariance', obligatory=True, dtype='float'), 
    'CNDOT_R':                      dict(cluster='data_covariance', obligatory=True, dtype='float'), 
    'CNDOT_T':                      dict(cluster='data_covariance', obligatory=True, dtype='float'), 
    'CNDOT_N':                      dict(cluster='data_covariance', obligatory=True, dtype='float'), 
    'CNDOT_RDOT':                   dict(cluster='data_covariance', obligatory=True, dtype='float'), 
    'CNDOT_TDOT':                   dict(cluster='data_covariance', obligatory=True, dtype='float'), 
    'CNDOT_NDOT':                   dict(cluster='data_covariance', obligatory=True, dtype='float'), 
    'CDRG_R':                       dict(cluster='data_covariance', obligatory=False, dtype='float'), 
    'CDRG_T':                       dict(cluster='data_covariance', obligatory=False, dtype='float'), 
    'CDRG_N':                       dict(cluster='data_covariance', obligatory=False, dtype='float'), 
    'CDRG_RDOT':                    dict(cluster='data_covariance', obligatory=False, dtype='float'), 
    'CDRG_TDOT':                    dict(cluster='data_covariance', obligatory=False, dtype='float'), 
    'CDRG_NDOT':                    dict(cluster='data_covariance', obligatory=False, dtype='float'), 
    'CDRG_DRG':                     dict(cluster='data_covariance', obligatory=False, dtype='float'), 
    'CSRP_R':                       dict(cluster='data_covariance', obligatory=False, dtype='float'), 
    'CSRP_T':                       dict(cluster='data_covariance', obligatory=False, dtype='float'), 
    'CSRP_N':                       dict(cluster='data_covariance', obligatory=False, dtype='float'), 
    'CSRP_RDOT':                    dict(cluster='data_covariance', obligatory=False, dtype='float'), 
    'CSRP_TDOT':                    dict(cluster='data_covariance', obligatory=False, dtype='float'), 
    'CSRP_NDOT':                    dict(cluster='data_covariance', obligatory=False, dtype='float'), 
    'CSRP_DRG':                     dict(cluster='data_covariance', obligatory=False, dtype='float'), 
    'CSRP_SRP':                     dict(cluster='data_covariance', obligatory=False, dtype='float'), 
    'CTHR_R':                       dict(cluster='data_covariance', obligatory=False, dtype='float'), 
    'CTHR_T':                       dict(cluster='data_covariance', obligatory=False, dtype='float'), 
    'CTHR_N':                       dict(cluster='data_covariance', obligatory=False, dtype='float'), 
    'CTHR_RDOT':                    dict(cluster='data_covariance', obligatory=False, dtype='float'), 
    'CTHR_TDOT':                    dict(cluster='data_covariance', obligatory=False, dtype='float'), 
    'CTHR_NDOT':                    dict(cluster='data_covariance', obligatory=False, dtype='float'), 
    'CTHR_DRG':                     dict(cluster='data_covariance', obligatory=False, dtype='float'), 
    'CTHR_SRP':                     dict(cluster='data_covariance', obligatory=False, dtype='float'), 
    'CTHR_THR':                     dict(cluster='data_covariance', obligatory=False, dtype='float')
}



#%% FUNCTION: get_features
def get_features(by_cluster:bool=True, only_names:bool = False, 
                 suffix:str=None, include_object_preffix:bool=False, 
                 **kwfilters) -> None:
    """Get features from the reference cdm_features dictionary.

    Args:
        by_cluster (bool, optional): Organise by cluster ('header', 
        'relative metadata', 'metadata', 'data'). Defaults to True.
        only_names (bool, optional): Return only feature labels. Defaults to 
        False.
        suffix (str, optional): Suffix to add to the feature labels. Defaults to 
        None.
        include_object_preffix (bool, optional): Add OBJECTID tag to the feature 
        label. Defaults to False.
    """


    if not isinstance(kwfilters, dict): return None

    # Initialize output dictionary.
    output = {}
    features = []

    # Iterate over all features that a CDM could contain.
    for feature, information in cdm_features.items():

        # Exclude feature by default
        exclude = False

        cluster = information['cluster']

        if by_cluster and not cluster in output.keys():
            output[cluster] = []

        # Run through all filtering criterias passed with kwargs
        for field, filter in kwfilters.items():

            # If filters passed through kwargs are not a list, convert value as 
            # a list.
            if not isinstance(filter, list): filter = [filter]

            # If a criteria 'field' has multiple conditions, check all 
            # parameters from feature match at least one of those conditions.
            exclude = True if information[field] not in filter else exclude

        # Go to the next feature if it shall be excluded.
        if exclude: continue

        if include_object_preffix and not (cluster.startswith('header') or cluster.startswith('relative')):
            for o in [1, 2]:
                features.append(f'OBJECT{o}_{feature}')
                output[cluster].append(f'OBJECT{o}_{feature}')
        else:
            features.append(feature)
            output[cluster].append(feature)

    if suffix is not None:
        output = {f'{k}_{suffix}':v for k,v in output.items()}

    return features if only_names else output

def dtype_conversion() -> dict:
    """Get dictionary with the conversion dtypes for pandas DataFrame according 
    to the CCSDS standards.

    Returns:
        dict: Dictionary with the mapping column:dtype according to the CCSDS 
        standards.
    """
    # Cast columns as correct dtypes
    convert_dict = dict()

    # Iterate over all objects
    for o in [1, 2]:

        # Iterate over all features in a CDM
        for feature, information in cdm_features.items():

            # Get the cluster of the CDM data.
            cluster = information['cluster']

            if (cluster.startswith('header') or \
                cluster.startswith('relative')) and o==1:
                xfeature = feature
            elif not (cluster.startswith('header') or \
                        cluster.startswith('relative')):
                xfeature = f'OBJECT{o}_{feature}'

            if xfeature in convert_dict.keys(): continue
            #if xfeature not in df.columns: continue
            convert_dict[xfeature] = information['dtype']
    return convert_dict


# Declare global variable containing all keys within every cluster in a CDM.
cdm_clusters = get_features()
cdm_clusters_obligatory = get_features(suffix='obligatory', 
                                       **dict(obligatory=True))
df_dtype_conversion = dtype_conversion()


if __name__ == "__main__":

    print(utils.format_json(get_features(suffix='obligatory', **dict(obligatory=True))))
    print(utils.format_json(get_features()))
