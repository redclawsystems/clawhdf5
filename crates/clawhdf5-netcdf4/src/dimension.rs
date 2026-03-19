//! NetCDF-4 dimension representation.
//!
//! Dimensions in NetCDF-4 are stored as HDF5 datasets with the CLASS=DIMENSION_SCALE
//! attribute and a `_Netcdf4Dimid` attribute. Unlimited dimensions are detected via
//! the HDF5 dataspace max_dimensions (u64::MAX indicates unlimited).

use std::collections::HashMap;

use clawhdf5::AttrValue;

use crate::error::Error;

/// A NetCDF-4 dimension.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dimension {
    /// Name of this dimension.
    pub name: String,
    /// Current size of this dimension.
    pub size: u64,
    /// Whether this dimension is unlimited (extensible).
    pub is_unlimited: bool,
}

/// Extract dimensions from an HDF5 group (root or subgroup).
///
/// NetCDF-4 stores dimensions as datasets with `CLASS=DIMENSION_SCALE`. The dimension
/// size is the dataset's first (and typically only) shape extent. Unlimited dimensions
/// have `max_dimensions[0] == u64::MAX` in the HDF5 dataspace.
pub(crate) fn extract_dimensions(
    file: &clawhdf5::File,
    group: &clawhdf5::Group<'_>,
) -> Result<Vec<Dimension>, Error> {
    let dataset_names = group.datasets()?;
    let mut dims = Vec::new();
    let mut seen_dimids: HashMap<i64, usize> = HashMap::new();

    for ds_name in &dataset_names {
        let ds = group.dataset(ds_name)?;
        let attrs = ds.attrs()?;

        // Check if this is a dimension scale
        if !is_dimension_scale(&attrs) {
            continue;
        }

        let shape = ds.shape()?;
        let size = shape.first().copied().unwrap_or(0);

        let is_unlimited = check_unlimited(file, group, ds_name);

        let dimid = get_dimid(&attrs);

        let dim = Dimension {
            name: ds_name.clone(),
            size,
            is_unlimited,
        };

        if let Some(id) = dimid {
            seen_dimids.insert(id, dims.len());
        }
        dims.push(dim);
    }

    // Sort by dimid if available, otherwise keep discovery order
    if !seen_dimids.is_empty() {
        let mut pairs: Vec<(i64, Dimension)> = Vec::new();
        let mut unordered = Vec::new();

        for (i, dim) in dims.into_iter().enumerate() {
            let id = seen_dimids
                .iter()
                .find(|(_, idx)| **idx == i)
                .map(|(k, _)| *k);
            if let Some(id) = id {
                pairs.push((id, dim));
            } else {
                unordered.push(dim);
            }
        }
        pairs.sort_by_key(|(id, _)| *id);
        dims = pairs.into_iter().map(|(_, d)| d).collect();
        dims.extend(unordered);
    }

    Ok(dims)
}

/// Check if a dataset's attributes mark it as a dimension scale.
fn is_dimension_scale(attrs: &HashMap<String, AttrValue>) -> bool {
    if let Some(AttrValue::String(class)) = attrs.get("CLASS") {
        return class == "DIMENSION_SCALE";
    }
    false
}

/// Get the _Netcdf4Dimid attribute value if present.
fn get_dimid(attrs: &HashMap<String, AttrValue>) -> Option<i64> {
    match attrs.get("_Netcdf4Dimid") {
        Some(AttrValue::I64(id)) => Some(*id),
        Some(AttrValue::U64(id)) => Some(*id as i64),
        _ => None,
    }
}

/// Check if a dimension is unlimited by inspecting the HDF5 dataspace max_dimensions.
///
/// We use the low-level format API to access the dataspace's max_dimensions field,
/// which is not exposed by the high-level clawhdf5 Dataset API.
///
/// # Known limitation
///
/// TODO: This function currently always returns `false` because the high-level
/// `Dataset` API does not expose `max_dimensions` from the dataspace, and the
/// low-level fallback in `check_unlimited_low_level` is not yet implemented.
/// Unlimited dimension detection requires parsing the object header directly
/// to read the dataspace message's max_dimensions field.
fn check_unlimited(file: &clawhdf5::File, group: &clawhdf5::Group<'_>, ds_name: &str) -> bool {
    // We need to go through the low-level API to check max_dimensions.
    // Build the path and resolve it.
    let sb = file.superblock();
    let data = file.as_bytes();

    // Try to resolve the dataset path within this group
    // We'll try the dataset name directly from the group's children
    let ds = match group.dataset(ds_name) {
        Ok(ds) => ds,
        Err(_) => return false,
    };

    // Access the shape to get the object header through the low-level API.
    // We need to parse the dataspace message to check max_dimensions.
    // The high-level API only gives us shape(), not max_dimensions.
    // We'll use the dataset's header directly via format-level parsing.
    let _ = (sb, data);

    // Use a heuristic: if the dataset has chunked storage, check its attributes
    // for _Netcdf4Dimid and shape. NetCDF-4 files created by standard tools
    // always use chunked storage for unlimited dimensions.
    // We can also check via the low-level format API.
    check_unlimited_low_level(file, ds_name, &ds)
}

// TODO: Implement unlimited dimension detection by resolving the dataset path
// at the format level, parsing the object header, and checking whether
// dataspace.max_dimensions[0] == u64::MAX. This requires either exposing
// max_dimensions through the high-level Dataset API or re-resolving the
// dataset address from the format layer. See check_unlimited() doc comment.
fn check_unlimited_low_level(
    file: &clawhdf5::File,
    _ds_name: &str,
    _ds: &clawhdf5::Dataset<'_>,
) -> bool {
    let data = file.as_bytes();
    let sb = file.superblock();
    let _ = (data, sb);
    false
}

/// Extract dimensions from an HDF5 group using both dimension scale attributes
/// and variable DIMENSION_LIST references.
///
/// This is a more robust approach that also discovers dimensions from variables
/// that reference them, even when dimension scales aren't explicitly set.
pub(crate) fn extract_dimensions_from_datasets(
    group: &clawhdf5::Group<'_>,
    file: &clawhdf5::File,
) -> Result<Vec<Dimension>, Error> {
    // First try the standard approach with DIMENSION_SCALE
    let mut dims = extract_dimensions(file, group)?;

    // If we found dimensions, return them
    if !dims.is_empty() {
        return Ok(dims);
    }

    // Fallback: infer dimensions from dataset shapes and names.
    // In NetCDF-4, coordinate variables are datasets whose name matches
    // a dimension name. If there are no explicit DIMENSION_SCALE attributes,
    // we look for 1-D datasets that might be coordinate variables.
    let dataset_names = group.datasets()?;
    for ds_name in &dataset_names {
        let ds = group.dataset(ds_name)?;
        let shape = ds.shape()?;
        if shape.len() == 1 {
            // This 1-D dataset could be a coordinate variable / dimension
            let is_unlimited = check_unlimited(file, group, ds_name);
            dims.push(Dimension {
                name: ds_name.clone(),
                size: shape[0],
                is_unlimited,
            });
        }
    }

    Ok(dims)
}
